"""A tiny agent. One tool: the terminal.

    TINY_AGENT_API_KEY=... python agent.py

Four parts, same shape as the model file: Model (the LLM behind an
OpenAI-compatible endpoint), Terminal (the tool), Transcript (every glyph and
color), and Agent (the think → act → observe loop that composes them).
"""

import json
import os
import select
import subprocess
import sys
import termios
import textwrap
import time
import tty
import urllib.error
import urllib.request

from rich.console import Console
from rich.text import Text

ASCII_LOGO = """
██╗    ████████╗██╗███╗   ██╗██╗   ██╗    ██████╗ ██╗    ██╗███████╗███╗   ██╗
╚██╗   ╚══██╔══╝██║████╗  ██║╚██╗ ██╔╝   ██╔═══██╗██║    ██║██╔════╝████╗  ██║
 ╚██╗     ██║   ██║██╔██╗ ██║ ╚████╔╝    ██║   ██║██║ █╗ ██║█████╗  ██╔██╗ ██║
 ██╔╝     ██║   ██║██║╚██╗██║  ╚██╔╝     ██║▄▄ ██║██║███╗██║██╔══╝  ██║╚██╗██║
██╔╝      ██║   ██║██║ ╚████║   ██║      ╚██████╔╝╚███╔███╔╝███████╗██║ ╚████║
╚═╝       ╚═╝   ╚═╝╚═╝  ╚═══╝   ╚═╝       ╚══▀▀═╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═══╝
"""

STARTING_HELP_TEXT = """\
• Enter anything
• Esc to interrupt, Ctrl+C to exit
• /verbose to toggle full command output
"""

SYSTEM = """You are a capable agent working from a terminal. The terminal is your only
tool — files, code, searches, everything happens through shell commands. Keep commands
short and non-interactive. When the task is done, answer in plain text with no markdown please. The user is talking to you via a terminal and markdowns will not render. Also, be concise."""

TOOL = {
    "type": "function",
    "function": {
        "name": "terminal",
        "description": "Run a shell command and see its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}

INTERRUPT_NOTE = (
    "(the user pressed Esc — they do not want this executed. "
    "Stop what you were doing and wait for their next message.)"
)

BASE_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
MODEL = "accounts/fireworks/models/kimi-k3"
API_KEY = os.environ.get("TINY_AGENT_API_KEY")
MAX_STEPS = 40
COMMAND_TIMEOUT = 120
OUTPUT_PREVIEW_LINES = 3

console = Console(highlight=False)


class EscWatch:
    """Puts the tty in cbreak for the turn so Esc keypresses can be polled."""

    def __enter__(self):
        self.enabled = sys.stdin.isatty()
        if self.enabled:
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def pressed(self):
        hit = False
        while self.enabled and select.select([sys.stdin], [], [], 0)[0]:
            if sys.stdin.read(1) == "\x1b":
                hit = True
        return hit

    def __exit__(self, *exc):
        if self.enabled:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


class Model:
    """The LLM behind an OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, base_url, model, key):
        self.base_url = base_url
        self.model = model
        self.key = key

    def step(self, messages, status):
        """One streamed step. Appends the reply to messages,
        returns (thoughts, text, [(call_id, command)])."""
        resp = self._request(messages)
        thoughts, text, calls = self._read_stream(resp, status)

        reply = {"role": "assistant", "content": text}
        if thoughts:
            reply["reasoning_content"] = thoughts
        if calls:
            reply["tool_calls"] = calls
        messages.append(reply)

        commands = [
            (call["id"], json.loads(call["function"]["arguments"])["command"])
            for call in calls
        ]
        return thoughts, text, commands

    def _request(self, messages):
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": [TOOL],
            "temperature": 0.3,
            "stream": True,
        }
        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.key}",
                # Fireworks' WAF rejects Python's default "Python-urllib/x" agent with 403
                "User-Agent": "tiny-qwen-agent",
            },
        )
        for attempt in range(4):
            try:
                return urllib.request.urlopen(req, timeout=300)
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and attempt < 3:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise

    def _read_stream(self, resp, status):
        """Accumulate SSE deltas into (thoughts, text, tool calls)."""
        thoughts, text, calls = "", "", []
        with resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                choices = json.loads(line[6:]).get("choices")
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("reasoning_content"):
                    if not thoughts:
                        status.update(Text("Thinking…", style="dim"))
                    thoughts += delta["reasoning_content"]
                if delta.get("content"):
                    text += delta["content"]
                for call in delta.get("tool_calls") or []:
                    if call.get("id"):
                        calls.append(
                            {
                                "id": call["id"],
                                "type": "function",
                                "function": {
                                    "name": call["function"]["name"],
                                    "arguments": "",
                                },
                            }
                        )
                    if (call.get("function") or {}).get("arguments"):
                        calls[-1]["function"]["arguments"] += call["function"]["arguments"]
        return thoughts, text, calls


class Terminal:
    """The one tool: run a shell command, polling for Esc."""

    def run(self, command, esc):
        """Returns command output, or None if the user interrupted."""
        proc = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        deadline = time.time() + COMMAND_TIMEOUT
        while True:
            try:
                output, _ = proc.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                if esc.pressed():
                    proc.kill()
                    proc.communicate()
                    return None
                if time.time() > deadline:
                    proc.kill()
                    output, _ = proc.communicate()
                    output = (output or "") + "\n(command timed out)"
                    break
        return output.strip()[:8000] or "(no output)"


class Transcript:
    """Every glyph, color, and indent in one place."""

    ACCENT = "#face0a"

    def banner(self):
        os.system("cls" if os.name == "nt" else "clear")
        console.print(Text(ASCII_LOGO, style=self.ACCENT))
        console.print(Text(STARTING_HELP_TEXT, style="dim"))

    def prompt(self):
        return console.input(f"[bold {self.ACCENT}]❯ [/]").strip()

    def spinner(self):
        waiting = Text("Waiting…", style="dim")
        return console.status(waiting, spinner="dots", spinner_style=self.ACCENT)

    def action(self, command):
        console.print(Text(f"$ {command}", style="dim"))

    def output(self, output, verbose):
        lines = output.splitlines()
        shown = lines if verbose else lines[:OUTPUT_PREVIEW_LINES]
        for i, line in enumerate(shown):
            connector = "  └ " if i == 0 else "    "
            console.print(Text(connector + line, style="dim"))
        hidden = len(lines) - len(shown)
        if hidden > 0:
            console.print(Text(f"    … +{hidden} lines (/verbose to see all)", style="dim"))
        console.print()

    def thoughts(self, thoughts):
        for line in self._hang(thoughts.strip()):
            console.print(Text("  " + line, style="dim italic"))
        console.print()

    def reply(self, answer):
        text = Text()
        text.append("• ", style="dim")
        text.append("\n  ".join(self._hang(answer)))
        console.print(text)
        console.print()

    def note(self, message):
        console.print(Text(message, style="dim"))
        console.print()

    def _hang(self, text, indent=2):
        """Wrap to console width minus indent; a flat list of lines."""
        width = max(console.width - indent, 20)
        lines = []
        for paragraph in text.splitlines() or [""]:
            lines.extend(textwrap.wrap(paragraph, width) or [""])
        return lines


class Agent:
    """think → act → observe, until the model answers in plain text."""

    def __init__(self):
        self.model = Model(BASE_URL, MODEL, API_KEY)
        self.terminal = Terminal()
        self.transcript = Transcript()
        self.messages = [{"role": "system", "content": SYSTEM}]
        self.verbose = False

    def turn(self, user_input):
        """Returns the final answer, or None if the user pressed Esc."""
        self.messages.append({"role": "user", "content": user_input})
        with EscWatch() as esc:
            for _ in range(MAX_STEPS):
                with self.transcript.spinner() as status:
                    thoughts, text, commands = self.model.step(self.messages, status)
                if self.verbose and thoughts:
                    self.transcript.thoughts(thoughts)
                if not commands:
                    return text
                if esc.pressed():
                    for call_id, _ in commands:
                        self._observe(call_id, INTERRUPT_NOTE)
                    return None
                if self._act(commands, esc):
                    return None
        return "(stopped: reached max steps)"

    def _act(self, commands, esc):
        """Run each command, printing receipts. Returns True if interrupted."""
        interrupted = False
        for call_id, command in commands:
            if interrupted:
                output = None
            else:
                self.transcript.action(command)
                output = self.terminal.run(command, esc)
            if output is None:
                interrupted = True
                self.transcript.output("(interrupted)", self.verbose)
                self._observe(call_id, INTERRUPT_NOTE)
            else:
                self.transcript.output(output, self.verbose)
                self._observe(call_id, output)
        return interrupted

    def _observe(self, call_id, output):
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": output})


def main():
    if not API_KEY:
        console.print("Set TINY_AGENT_API_KEY to use the agent.", style="red")
        return
    agent = Agent()
    agent.transcript.banner()
    try:
        while True:
            user_input = agent.transcript.prompt()
            if not user_input:
                continue
            if user_input == "/verbose":
                agent.verbose = not agent.verbose
                agent.transcript.note(f"verbose {'on' if agent.verbose else 'off'}")
                continue

            console.print()
            try:
                answer = agent.turn(user_input)
            except Exception as e:
                console.print(f"Error: {e}", style="red")
                if agent.messages[-1]["role"] == "user":
                    agent.messages.pop()
                continue
            if answer is None:
                agent.transcript.note("interrupted — your turn")
            else:
                agent.transcript.reply(answer)
    except (KeyboardInterrupt, EOFError):
        console.print("\nGoodbye!")


if __name__ == "__main__":
    main()
