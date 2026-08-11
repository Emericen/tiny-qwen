import os
import re
import io
import time
import warnings
import logging
import torch
import traceback
from contextlib import redirect_stderr

from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars
from rich.console import Console
from rich.text import Text

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

disable_progress_bars()

from model.processor import Processor
from model.model import Qwen3_5

ASCII_LOGO = """
██╗    ████████╗██╗███╗   ██╗██╗   ██╗    ██████╗ ██╗    ██╗███████╗███╗   ██╗
╚██╗   ╚══██╔══╝██║████╗  ██║╚██╗ ██╔╝   ██╔═══██╗██║    ██║██╔════╝████╗  ██║
 ╚██╗     ██║   ██║██╔██╗ ██║ ╚████╔╝    ██║   ██║██║ █╗ ██║█████╗  ██╔██╗ ██║
 ██╔╝     ██║   ██║██║╚██╗██║  ╚██╔╝     ██║▄▄ ██║██║███╗██║██╔══╝  ██║╚██╗██║
██╔╝      ██║   ██║██║ ╚████║   ██║      ╚██████╔╝╚███╔███╔╝███████╗██║ ╚████║
╚═╝       ╚═╝   ╚═╝╚═╝  ╚═══╝   ╚═╝       ╚══▀▀═╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═══╝
"""

STARTING_HELP_TEXT = """
Welcome to Tiny-Qwen Interactive Chat!

Tips:
1. /help for more information.
2. /exit or Ctrl+C to exit.
"""


HELP_TEXT = """
Available commands:
/help - Show this help message
/exit - Exit the application

Use @relative/path/to/image.jpg to include images in your messages.
"""

ALL_MODELS = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B",
]

console = Console(highlight=False)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore", message=r"The safetensors archive passed at .* does not contain metadata.*"
)
warnings.filterwarnings(
    "ignore", message=r"Some weights of the model checkpoint at .* were not used when initializing .*"
)


def resolve_stop_tokens(tokenizer):
    # Qwen3.5 special tokens
    token_names = ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]
    stop_tokens = []
    for name in token_names:
        token_id = tokenizer.token_to_id(name)
        if token_id is not None:
            stop_tokens.append(token_id)
    return stop_tokens


def parse_user_input(text):
    image_pattern = r"@([^\s]+\.(?:jpg|jpeg|png|gif|webp))"
    matches = list(re.finditer(image_pattern, text, re.IGNORECASE))

    if not matches:
        return [{"role": "user", "content": [{"type": "text", "text": text}]}]

    content = []
    last_end = 0

    for match in matches:
        if match.start() > last_end:
            text_part = text[last_end : match.start()].strip()
            if text_part:
                content.append({"type": "text", "text": text_part})

        image_path = match.group(1)
        if os.path.exists(image_path):
            content.append({"type": "image", "image": image_path})
        else:
            console.print(f"Warning: Image not found: {image_path}", style="yellow")

        last_end = match.end()

    if last_end < len(text):
        remaining_text = text[last_end:].strip()
        if remaining_text:
            content.append({"type": "text", "text": remaining_text})

    return [{"role": "user", "content": content}]


def generate_local_response(messages, model, processor, max_tokens=2048):
    device = next(model.parameters()).device
    inputs = processor(
        messages, add_generation_prompt=True, enable_thinking=False, device=device
    )

    stop_tokens = resolve_stop_tokens(processor.tokenizer)
    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "max_new_tokens": max_tokens,
        "stop_tokens": stop_tokens,
    }

    if inputs["pixels"] is not None:
        generation_kwargs["pixels"] = inputs["pixels"]
    if inputs["d_image"] is not None:
        generation_kwargs["d_image"] = inputs["d_image"]

    token_generator = model.generate_stream(**generation_kwargs)
    generated_tokens = []
    previous_text = ""
    start_time = None
    for token_id in token_generator:
        if start_time is None:
            start_time = time.time()
        generated_tokens.append(token_id)
        current_text = processor.tokenizer.decode(generated_tokens)
        new_text = current_text[len(previous_text) :]
        if new_text:
            previous_text = current_text
            yield new_text

    # Calculate and return generation stats
    end_time = time.time()
    num_tokens = len(generated_tokens)
    elapsed = end_time - start_time if start_time else 0
    tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0
    yield {"stats": {"tokens": num_tokens, "elapsed": elapsed, "tok_per_sec": tokens_per_sec}}


def main():
    try:
        # clear terminal
        os.system("cls" if os.name == "nt" else "clear")

        # show logo and help message
        yellow_logo = Text(ASCII_LOGO, style="#face0a")
        console.print(yellow_logo)
        console.print(STARTING_HELP_TEXT)

        # select model variant
        for i, variant in enumerate(ALL_MODELS, 1):
            console.print(f"  [#face0a]{i}[/] {variant}")
        choice = console.input("\n[bold]Select model variant [1]: [/]").strip() or "1"
        if not choice.isdigit() or not 1 <= int(choice) <= len(ALL_MODELS):
            return
        selected_model_variant = ALL_MODELS[int(choice) - 1]

        # load model
        hf_repo_id = selected_model_variant
        with console.status(f"[bold #face0a]Loading {hf_repo_id}...", spinner="dots"):
            try:
                with redirect_stderr(io.StringIO()):
                    weights_path = snapshot_download(repo_id=hf_repo_id, cache_dir=".cache")
                    processor = Processor.from_pretrained(hf_repo_id)
                    model = Qwen3_5.from_pretrained(
                        weights_path=weights_path, device_map="auto"
                    )
                model.eval()
                model = torch.compile(model)
            except Exception as e:
                console.print(f"Failed to load model: {e}")
                return

        if model is None or processor is None:
            console.print("Failed to initialize processor. Exiting...", style="red")
            return

        # start REPL
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "The assistant anwsers user's question concisely and accurately, ideally in a paragraph or two if not a few sentences. Since the assistant is interacting with the user in a CLI, it will only respond in plain text, avoiding emoji or markdown unless specifically requested.",
                    }
                ],
            }
        ]
        while True:
            user_input = console.input("[bold]USER: [/]")

            if user_input is None:
                console.print("Goodbye!")
                break

            user_input = user_input.strip()

            if user_input == "/exit":
                console.print("Goodbye!")
                break
            elif user_input == "/help":
                console.print(HELP_TEXT)
                continue
            elif not user_input:
                continue

            current_messages = parse_user_input(user_input)
            messages.extend(current_messages)

            try:
                response_segments = []
                print("QWEN: ", end="", flush=True)
                stats = None
                for segment in generate_local_response(messages, model, processor):
                    if isinstance(segment, dict) and "stats" in segment:
                        stats = segment["stats"]
                    else:
                        print(segment, end="", flush=True)
                        response_segments.append(segment)
                response = "".join(response_segments)
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": response}],
                    }
                )
                print()
                if stats:
                    console.print(
                        f"[dim]Generated {stats['tokens']} tokens in {stats['elapsed']:.2f}s "
                        f"({stats['tok_per_sec']:.2f} tok/s)[/dim]"
                    )
            except Exception as e:
                console.print(f"Error generating response: {e}", style="red")
                console.print(traceback.format_exc(), style="red")
                if messages and messages[-1]["role"] == "user":
                    messages.pop()

    except KeyboardInterrupt:
        console.print("\nGoodbye!")
    except Exception as e:
        console.print(f"Error: {e}", style="red")
        raise


if __name__ == "__main__":
    main()
