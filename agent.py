"""A tiny agent runtime. One tool: the terminal.

    python agent.py                          # interactive, remote model
    python agent.py --local Qwen/Qwen3.5-4B  # interactive, local tiny-qwen model
    python agent.py -p "do something"        # headless: run one task, print answer, exit
    python agent.py -p "task" --yolo         # no confirmation before running commands

The agent can spawn sub-agents by running this file with -p in its own terminal —
nesting is not a feature, it falls out of the terminal being a tool.

Remote mode talks to any OpenAI-compatible endpoint (stdlib HTTP, no client library;
set TINY_AGENT_API_KEY). Local mode runs a tiny-qwen model and parses Qwen's native
<tool_call> format straight out of the generated text.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

from rich.console import Console

console = Console(highlight=False)
SELF = f"{sys.executable} {os.path.abspath(__file__)}"

SYSTEM = f"""You are a capable agent working from a terminal. The terminal is your only
tool — files, code, searches, everything happens through shell commands. Keep commands
short and non-interactive. To delegate, spawn a sub-agent with:
{SELF} -p "<subtask>"
When the task is done, answer in plain text without calling the tool. Be concise."""

LOCAL_SYSTEM = SYSTEM + """

# Tools

You may call the terminal tool. To call it, reply with exactly:
<tool_call>
{"name": "terminal", "arguments": {"command": "<shell command>"}}
</tool_call>
The result will come back inside <tool_response></tool_response>."""

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

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


# ---------------------------------------------------------------- backends


class RemoteModel:
    """Any OpenAI-compatible /chat/completions endpoint, via stdlib HTTP."""

    def __init__(self, args):
        self.args = args

    def step(self, messages):
        """messages in, (text, [commands]) out."""
        payload = {
            "model": self.args.model,
            "messages": messages,
            "tools": [TOOL],
            "temperature": 0.3,
        }
        req = urllib.request.Request(
            self.args.base_url.rstrip("/") + "/chat/completions",
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

    @staticmethod
    def tool_result(messages, call_id, output):
        messages.append({"role": "tool", "tool_call_id": call_id, "content": output})


class LocalModel:
    """A tiny-qwen model. Tool calls are parsed from the generated text."""

    def __init__(self, args):
        import torch
        from huggingface_hub import snapshot_download
        from model.processor import Processor
        from model.model import Qwen3_5

        with console.status(f"[bold #face0a]Loading {args.local}...", spinner="dots"):
            path = snapshot_download(repo_id=args.local, cache_dir=".cache")
            self.processor = Processor.from_pretrained(args.local)
            self.model = Qwen3_5.from_pretrained(
                weights_path=path,
                device_map={"": "mps" if torch.backends.mps.is_available() else "cpu"},
            )
            self.model.eval()
        self.device = next(self.model.parameters()).device
        self.stops = [
            t
            for t in (self.processor.tokenizer.token_to_id(n) for n in ("<|im_end|>", "<|endoftext|>"))
            if t is not None
        ]
        self.max_tokens = args.max_tokens

    def step(self, messages):
        blocks = [
            {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
            for m in messages
        ]
        inputs = self.processor(
            blocks, add_generation_prompt=True, enable_thinking=False, device=self.device
        )
        token_ids = []
        for token_id in self.model.generate_stream(
            input_ids=inputs["input_ids"],
            max_new_tokens=self.max_tokens,
            stop_tokens=self.stops,
        ):
            token_ids.append(token_id)
        text = self.processor.tokenizer.decode(token_ids).strip()
        messages.append({"role": "assistant", "content": text})
        commands = []
        for i, match in enumerate(TOOL_CALL_RE.finditer(text)):
            try:
                commands.append((f"local-{i}", json.loads(match.group(1))["arguments"]["command"]))
            except (json.JSONDecodeError, KeyError):
                pass
        clean = TOOL_CALL_RE.sub("", text).strip()
        return clean, commands

    @staticmethod
    def tool_result(messages, call_id, output):
        messages.append({"role": "user", "content": f"<tool_response>\n{output}\n</tool_response>"})


# ---------------------------------------------------------------- the loop


def run_command(command, args):
    if not args.yolo:
        console.print(f"[bold #face0a]run?[/] [dim]{command}[/]", end=" ")
        if input("[y/N] ").strip().lower() != "y":
            return "(user declined to run this command)"
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=args.timeout
    )
    output = (result.stdout + result.stderr).strip()
    return output[:8000] or "(no output)"


def agent_turn(backend, messages, args):
    """Run the model until it stops calling the tool. Returns its final text."""
    for _ in range(args.max_steps):
        text, commands = backend.step(messages)
        if not commands:
            return text
        for call_id, command in commands:
            console.print(f"[dim]$ {command}[/]")
            output = run_command(command, args)
            if args.verbose and output:
                console.print(f"[dim]{output[:500]}[/]")
            backend.tool_result(messages, call_id, output)
    return "(stopped: reached max steps)"


def save_session(messages, path):
    with open(path, "w") as f:
        for m in messages:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="a tiny agent with one tool: the terminal")
    parser.add_argument("-p", "--print", dest="task", help="run one task headlessly and exit")
    parser.add_argument("--local", help="run a local tiny-qwen model, e.g. Qwen/Qwen3.5-4B")
    parser.add_argument("--model", default="kimi-k3", help="remote model name")
    parser.add_argument("--base-url", default="https://api.moonshot.cn/v1")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=1024, help="local generation cap")
    parser.add_argument("--timeout", type=int, default=120, help="per-command timeout")
    parser.add_argument("--yolo", action="store_true", help="run commands without confirming")
    parser.add_argument("--verbose", action="store_true", help="show command output")
    parser.add_argument("--session", help="write the conversation to this jsonl file")
    args = parser.parse_args()

    backend = LocalModel(args) if args.local else RemoteModel(args)
    system = LOCAL_SYSTEM if args.local else SYSTEM
    messages = [{"role": "system", "content": system}]

    if args.task:
        messages.append({"role": "user", "content": args.task})
        answer = agent_turn(backend, messages, args)
        if args.session:
            save_session(messages, args.session)
        print(answer)
        return

    console.print("[bold #face0a]tiny agent[/] — one tool: the terminal. /exit to quit.\n")
    while True:
        try:
            task = console.input("[bold #face0a]› [/]").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not task or task == "/exit":
            break
        messages.append({"role": "user", "content": task})
        answer = agent_turn(backend, messages, args)
        console.print(answer)
        if args.session:
            save_session(messages, args.session)


if __name__ == "__main__":
    main()
