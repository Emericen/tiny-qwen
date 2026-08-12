"""A tiny agent. One tool: the terminal.

    python agent.py

Talks to any OpenAI-compatible /chat/completions endpoint via stdlib HTTP — no
client library. Configure with environment variables:

    TINY_AGENT_API_KEY    required
    TINY_AGENT_BASE_URL   default https://api.moonshot.cn/v1
    TINY_AGENT_MODEL      default kimi-k3
"""

import json
import os
import subprocess
import time
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

STARTING_HELP_TEXT = """
Welcome to Tiny-Qwen Agent!

Tips:
1. The agent has one tool: the terminal. You approve every command.
2. /exit or Ctrl+C to exit.
"""

BASE_URL = os.environ.get("TINY_AGENT_BASE_URL", "https://api.moonshot.cn/v1")
MODEL = os.environ.get("TINY_AGENT_MODEL", "kimi-k3")
MAX_STEPS = 40
COMMAND_TIMEOUT = 120

SYSTEM = """You are a capable agent working from a terminal. The terminal is your only
tool — files, code, searches, everything happens through shell commands. Keep commands
short and non-interactive. When the task is done, answer in plain text without calling
the tool. Be concise."""

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

console = Console(highlight=False)


def chat(messages):
    """One model step. Appends the reply to messages, returns (text, [(id, command)])."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": [TOOL],
        "temperature": 0.3,
    }
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['TINY_AGENT_API_KEY']}",
        },
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                reply = json.load(resp)["choices"][0]["message"]
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            raise
    messages.append(reply)
    commands = []
    for call in reply.get("tool_calls") or []:
        commands.append((call["id"], json.loads(call["function"]["arguments"])["command"]))
    return reply.get("content") or "", commands


def run_command(command):
    console.print(f"[bold #face0a]run?[/] [dim]{command}[/]", end=" ")
    if input("[y/N] ").strip().lower() != "y":
        return "(user declined to run this command)"
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=COMMAND_TIMEOUT
    )
    output = (result.stdout + result.stderr).strip()
    return output[:8000] or "(no output)"


def agent_turn(messages):
    """Run the model until it stops calling the tool. Returns its final text."""
    for _ in range(MAX_STEPS):
        text, commands = chat(messages)
        if not commands:
            return text
        for call_id, command in commands:
            console.print(f"[dim]$ {command}[/]")
            output = run_command(command)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": output})
    return "(stopped: reached max steps)"


def main():
    try:
        os.system("cls" if os.name == "nt" else "clear")
        console.print(Text(ASCII_LOGO, style="#face0a"))
        console.print(STARTING_HELP_TEXT)

        if "TINY_AGENT_API_KEY" not in os.environ:
            console.print("Set TINY_AGENT_API_KEY to use the agent.", style="red")
            return

        messages = [{"role": "system", "content": SYSTEM}]
        while True:
            user_input = console.input("[bold]USER: [/]").strip()

            if user_input == "/exit":
                console.print("Goodbye!")
                break
            elif not user_input:
                continue

            messages.append({"role": "user", "content": user_input})
            try:
                answer = agent_turn(messages)
                console.print(f"QWEN: {answer}")
            except Exception as e:
                console.print(f"Error: {e}", style="red")
                if messages and messages[-1]["role"] == "user":
                    messages.pop()

    except (KeyboardInterrupt, EOFError):
        console.print("\nGoodbye!")


if __name__ == "__main__":
    main()
