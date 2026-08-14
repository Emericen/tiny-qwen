"""A tiny agent. One tool: the terminal.

    python run.py                           # small default model, downloads on first run
    python run.py Qwen/Qwen3.8-27B          # any HF repo id, fetched on demand
    python run.py Qwen/Qwen3.8-27B --bits 4 # quantize after download, then run
    python run.py weights/Qwen3.8-27B-int4  # any local dir, e.g. quantized
    python run.py --url https://api.fireworks.ai/inference/v1/chat/completions \
                    --api-key ... --model accounts/fireworks/models/kimi-k3

By default the model runs locally from weights/. The remote flags bypass the
local path entirely and talk to any OpenAI-compatible endpoint via stdlib HTTP.

Transcript style: your turns start with a yellow ❯, everything the agent does
(commands, output) is dim, everything it says is plain text behind a dim
bullet. Esc interrupts a running turn; /verbose shows thoughts and full
command output. Mention images inline as @path/to/image.jpg (local mode).
"""

import argparse
import json
import os
import re
import readline  # noqa: F401 — gives the input prompt arrows and history
import select
import subprocess
import sys
import termios
import textwrap
import time
import tty
import urllib.error
import urllib.request
from pathlib import Path

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
• Enter anything; @path/to/image.jpg to show the model an image
• Esc to interrupt, Ctrl+C to exit
• /verbose to toggle thoughts and full command output
"""

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"  # small, so first run succeeds on any machine
WEIGHTS_DIR = Path("weights")
HUB = "https://huggingface.co"

MAX_STEPS = 40
MAX_NEW_TOKENS = 1024
COMMAND_TIMEOUT = 120
OUTPUT_PREVIEW_LINES = 3

INTERRUPT_NOTE = (
    "(the user pressed Esc — they do not want this executed. "
    "Stop what you were doing and wait for their next message.)"
)

SYSTEM = """You are a capable agent working from a terminal. The terminal is your only
tool — files, code, searches, everything happens through shell commands. Keep commands
short and non-interactive. One extra built-in exists: `view <path/to/image>` shows you
an image file so you can actually see it. When the task is done, answer in plain text with no markdown please. The user is talking to you via a terminal and markdowns will not render. Also, be concise."""

# The exact tool-section wording Qwen models are trained on — small models
# only emit well-formed <tool_call> blocks when the prompt matches training.
LOCAL_SYSTEM = SYSTEM + """

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "terminal", "description": "Run a shell command and see its output.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

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
IMAGE_RE = re.compile(r"@([^\s]+\.(?:jpg|jpeg|png|gif|webp))", re.I)
VIEW_RE = re.compile(r"^\s*view\s+(\S+\.(?:jpg|jpeg|png|gif|webp))\s*$", re.I)

console = Console(highlight=False)
verbose = False


# ---------------------------------------------------------------- download

def download(repo, dest):
    """Fetch a model repo into dest/ using plain HTTPS. A file only ever
    appears under its real name once it is whole — partial downloads live
    under .part and are overwritten on retry, so anything in weights/ with a
    real name is complete."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{HUB}/api/models/{repo}/tree/main") as r:
        files = [(f["path"], f["size"]) for f in json.load(r) if f["type"] == "file"]

    DIM, RESET, CLEAR = "\033[2m", "\033[0m", "\033[K"
    for name, size in files:
        target = dest / name
        if target.exists() and target.stat().st_size == size:
            continue

        def progress(blocks, block_size, total, name=name):
            done = min(blocks * block_size, max(total, 1))
            print(f"\r{DIM}  {name}  {done / 1e6:.1f} / {total / 1e6:.1f} MB{RESET}{CLEAR}",
                  end="", flush=True)

        tmp = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(f"{HUB}/{repo}/resolve/main/{name}", tmp, progress)
        tmp.rename(target)
    print(f"\r{DIM}  {repo} ready{RESET}{CLEAR}")
    return dest


def resolve_model(target):
    """An explicit local directory is used as-is; anything else is a HF repo
    id, synced through download() — which skips complete files, so a finished
    model costs one listing request and an interrupted one resumes."""
    if (Path(target) / "config.json").exists():
        return Path(target)
    local = WEIGHTS_DIR / target.split("/")[-1]
    try:
        return download(target, local)
    except urllib.error.URLError:
        if (local / "config.json").exists():
            console.print(Text("offline — using existing local files", style="dim"))
            return local
        raise


def convert(src_dir, bits):
    """Stream-quantize a checkpoint directory into <src>-int<bits>. Tensors are
    read one at a time, so peak memory is one tensor — a 27B converts on a
    laptop. The output is self-contained and loads through QuantLinear."""
    import shutil
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from tiny_qwen.model import Model, quantize_int8, quantize_int4, GROUP

    src_dir = Path(src_dir)
    out = src_dir.parent / f"{src_dir.name}-int{bits}"
    if (out / "quant.json").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "preprocessor_config.json", "generation_config.json"):
        if (src_dir / name).exists():
            shutil.copy(src_dir / name, out / name)

    quantize = quantize_int8 if bits == 8 else quantize_int4
    quantized = []
    shard, shard_bytes, shard_idx = {}, 0, 0

    def flush():
        nonlocal shard, shard_bytes, shard_idx
        if shard:
            save_file(shard, str(out / f"model-{shard_idx:03d}.safetensors"))
            shard, shard_bytes, shard_idx = {}, 0, shard_idx + 1

    console.print(Text(f"quantizing to int{bits} …", style="dim"))
    for path in sorted(src_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                name = Model._rename(key)
                if name is None:
                    continue
                tensor = f.get_tensor(key)
                if tensor.dtype == torch.float32:
                    tensor = tensor.to(torch.bfloat16)
                is_linear_weight = (
                    tensor.ndim == 2
                    and name.startswith("layers.")
                    and name.endswith(".weight")
                    and tensor.shape[1] % GROUP == 0
                )
                if is_linear_weight:
                    q, scale = quantize(tensor)
                    shard[name.replace(".weight", ".qweight")] = q
                    shard[name.replace(".weight", ".scale")] = scale
                    shard_bytes += q.nbytes + scale.nbytes
                    quantized.append(name)
                else:
                    shard[name] = tensor
                    shard_bytes += tensor.nbytes
                if shard_bytes > 2_000_000_000:
                    flush()
    flush()
    (out / "quant.json").write_text(json.dumps({"bits": bits, "quantized": quantized}))
    console.print(Text(f"  {out} ready", style="dim"))
    return out


# ---------------------------------------------------------------- esc watch

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


# ---------------------------------------------------------------- backends

class LocalModel:
    """A model from weights/, spoken to through the processor's chat format.
    Tool calls are parsed from the generated text (Qwen's native format)."""

    def __init__(self, weights_path):
        from tiny_qwen import Model
        from tiny_qwen import Processor

        with console.status(Text(f"Loading {weights_path} …", style="dim"),
                            spinner="dots", spinner_style="#face0a"):
            self.processor = Processor.from_pretrained(weights_path)
            self.model = Model.from_pretrained(weights_path)
        self.device = next(self.model.parameters()).device
        self.messages = [
            {"role": "system", "content": [{"type": "text", "text": LOCAL_SYSTEM}]}
        ]

    def add_user(self, text):
        blocks, last = [], 0
        for match in IMAGE_RE.finditer(text):
            before = text[last:match.start()].strip()
            if before:
                blocks.append({"type": "text", "text": before})
            if os.path.exists(match.group(1)):
                blocks.append({"type": "image", "image": match.group(1)})
            else:
                console.print(Text(f"image not found: {match.group(1)}", style="dim"))
            last = match.end()
        rest = text[last:].strip()
        if rest or not blocks:
            blocks.append({"type": "text", "text": rest or text})
        self.messages.append({"role": "user", "content": blocks})

    def step(self, status):
        inputs = self.processor(
            self.messages, add_generation_prompt=True, enable_thinking=True,
            device=self.device,
        )
        status.update(Text("Thinking…", style="dim"))
        token_ids = []
        for token_id in self.model.generate_stream(
            input_ids=inputs["input_ids"],
            pixels=inputs["pixels"],
            d_image=inputs["d_image"],
            position_ids=inputs["position_ids"],
            max_new_tokens=MAX_NEW_TOKENS,
            stop_tokens=self.processor.stop_tokens,
        ):
            token_ids.append(token_id)
        text = self.processor.tokenizer.decode(token_ids).strip()

        # thinking precedes the reply; keep it out of the message history
        # (Qwen's own template drops past turns' thinking too)
        thoughts = ""
        if "</think>" in text:
            thoughts, text = text.split("</think>", 1)
            text = text.strip()
        self.messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": text}]}
        )
        commands = []
        for i, match in enumerate(TOOL_CALL_RE.finditer(text)):
            try:
                commands.append((f"local-{i}", json.loads(match.group(1))["arguments"]["command"]))
            except (json.JSONDecodeError, KeyError):
                pass
        return thoughts.strip(), TOOL_CALL_RE.sub("", text).strip(), commands

    def add_tool_result(self, call_id, output):
        self.messages.append(
            {"role": "tool", "content": [{"type": "text", "text": output}]}
        )

    def add_image_result(self, call_id, path):
        self.messages.append(
            {"role": "tool", "content": [
                {"type": "text", "text": "here is the image:"},
                {"type": "image", "image": path},
            ]}
        )


class RemoteModel:
    """Any OpenAI-compatible /chat/completions endpoint, via stdlib HTTP,
    streamed. The spinner reads "Waiting…" until the first token arrives,
    then "Thinking…" while reasoning tokens stream in."""

    def __init__(self, url, api_key, model):
        self.url = url
        self.api_key = api_key
        self.model = model
        self.messages = [{"role": "system", "content": SYSTEM}]

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text})

    def _request(self):
        payload = {
            "model": self.model,
            "messages": self.messages,
            "tools": [TOOL],
            "temperature": 0.3,
            "stream": True,
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                # some providers' WAFs reject Python's default agent with 403
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

    def step(self, status):
        thoughts, text, calls = "", "", []
        with self._request() as resp:
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

        reply = {"role": "assistant", "content": text}
        if thoughts:
            reply["reasoning_content"] = thoughts
        if calls:
            reply["tool_calls"] = calls
        self.messages.append(reply)
        commands = [
            (call["id"], json.loads(call["function"]["arguments"])["command"])
            for call in calls
        ]
        return thoughts, text, commands

    def add_tool_result(self, call_id, output):
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": output})

    def add_image_result(self, call_id, path):
        import base64
        mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        data = base64.b64encode(open(path, "rb").read()).decode()
        self.messages.append({"role": "tool", "tool_call_id": call_id,
                              "content": "(the image follows in the next message)"})
        self.messages.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}
        ]})


# ---------------------------------------------------------------- the loop

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

    def output(self, output):
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

    def __init__(self, backend):
        self.backend = backend
        self.terminal = Terminal()
        self.transcript = Transcript()

    def turn(self, user_input):
        """Returns the final answer, or None if the user pressed Esc."""
        self.backend.add_user(user_input)
        with EscWatch() as esc:
            for _ in range(MAX_STEPS):
                with self.transcript.spinner() as status:
                    thoughts, text, commands = self.backend.step(status)
                if verbose and thoughts:
                    self.transcript.thoughts(thoughts)
                if not commands:
                    return text
                if esc.pressed():
                    for call_id, _ in commands:
                        self.backend.add_tool_result(call_id, INTERRUPT_NOTE)
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
                seen = VIEW_RE.match(command)
                if seen:
                    path = os.path.expanduser(seen.group(1))
                    if os.path.exists(path):
                        self.backend.add_image_result(call_id, path)
                        self.transcript.output("(image shown to the model)")
                    else:
                        self.backend.add_tool_result(call_id, f"(no such image: {path})")
                        self.transcript.output(f"(no such image: {path})")
                    continue
                output = self.terminal.run(command, esc)
            if output is None:
                interrupted = True
                self.transcript.output("(interrupted)")
                self.backend.add_tool_result(call_id, INTERRUPT_NOTE)
            else:
                self.transcript.output(output)
                self.backend.add_tool_result(call_id, output)
        return interrupted


def main():
    global verbose
    parser = argparse.ArgumentParser(description="a tiny agent with one tool: the terminal")
    parser.add_argument("target", nargs="?", default=DEFAULT_MODEL,
                        help="local weights dir or HF repo id (downloaded on demand)")
    parser.add_argument("--url", help="OpenAI-compatible /chat/completions endpoint; "
                                      "with --api-key and --model, bypasses the local path")
    parser.add_argument("--api-key", default=os.environ.get("TINY_AGENT_API_KEY"))
    parser.add_argument("--model", help="remote model name (remote mode only)")
    parser.add_argument("--bits", type=int, choices=(4, 8),
                        help="quantize the model after download, then run it")
    args = parser.parse_args()

    agent = Agent(backend=None)
    agent.transcript.banner()

    try:
        if args.url:
            if not (args.api_key and args.model):
                console.print("Remote mode needs --url, --api-key, and --model.", style="red")
                return
            agent.backend = RemoteModel(args.url, args.api_key, args.model)
        else:
            weights_path = resolve_model(args.target)
            if args.bits:
                weights_path = convert(weights_path, args.bits)
            agent.backend = LocalModel(weights_path)
    except KeyboardInterrupt:
        print()
        agent.transcript.note("interrupted — finished files are kept; run again to resume")
        return

    try:
        while True:
            user_input = agent.transcript.prompt()

            if not user_input:
                continue
            if user_input == "/verbose":
                verbose = not verbose
                agent.transcript.note(f"verbose {'on' if verbose else 'off'}")
                continue

            console.print()
            try:
                answer = agent.turn(user_input)
            except Exception as e:
                console.print(f"Error: {e}", style="red")
                continue
            if answer is None:
                agent.transcript.note("interrupted — your turn")
            else:
                agent.transcript.reply(answer)

    except (KeyboardInterrupt, EOFError):
        console.print("\nGoodbye!")


if __name__ == "__main__":
    main()
