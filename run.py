import os
import re
import torch

import typer
import questionary
from questionary import Choice, Style
from rich.console import Console
from rich.text import Text

# Disable HuggingFace progress bars globally
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from model.processor import Processor
from model.model import Qwen2VL, Qwen3, Qwen3MoE

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

For Qwen2.5-VL, use @relative/path/to/image.jpg to include images in your messages.
"""

# Mapping of all models: generation -> variant -> HF repo id
ALL_MODELS = {
    "Qwen3": {
        "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
        "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
        "Qwen3-4B": "Qwen/Qwen3-4B",
        "Qwen3-8B": "Qwen/Qwen3-8B",
        "Qwen3-14B": "Qwen/Qwen3-14B",
        "Qwen3-32B": "Qwen/Qwen3-32B",
        "Qwen3-4B-Instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen3-30B-A3B-Instruct-2507": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen3-235B-A22B-Instruct-2507": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "Qwen3-4B-Thinking-2507": "Qwen/Qwen3-4B-Thinking-2507",
        "Qwen3-30B-A3B-Thinking-2507": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "Qwen3-235B-A22B-Thinking-2507": "Qwen/Qwen3-235B-A22B-Thinking-2507",
    },
    "Qwen2.5-VL": {
        "Qwen2.5-VL-3B-Instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen2.5-VL-32B-Instruct": "Qwen/Qwen2.5-VL-32B-Instruct",
        "Qwen2.5-VL-72B-Instruct": "Qwen/Qwen2.5-VL-72B-Instruct",
    },
}

REPO_ID_TO_MODEL_CLASS = {
    "Qwen/Qwen2.5-VL-3B-Instruct": Qwen2VL,
    "Qwen/Qwen2.5-VL-7B-Instruct": Qwen2VL,
    "Qwen/Qwen2.5-VL-32B-Instruct": Qwen2VL,
    "Qwen/Qwen2.5-VL-72B-Instruct": Qwen2VL,
    "Qwen/Qwen3-0.6B": Qwen3MoE,
    "Qwen/Qwen3-1.7B": Qwen3,
    "Qwen/Qwen3-4B": Qwen3MoE,
    "Qwen/Qwen3-8B": Qwen3,
    "Qwen/Qwen3-14B": Qwen3,
    "Qwen/Qwen3-32B": Qwen3,
    "Qwen/Qwen3-4B-Instruct-2507": Qwen3MoE,
    "Qwen/Qwen3-30B-A3B-Instruct-2507": Qwen3MoE,
    "Qwen/Qwen3-235B-A22B-Instruct-2507": Qwen3MoE,
    "Qwen/Qwen3-4B-Thinking-2507": Qwen3MoE,
    "Qwen/Qwen3-30B-A3B-Thinking-2507": Qwen3MoE,
    "Qwen/Qwen3-235B-A22B-Thinking-2507": Qwen3MoE,
}

STYLE = Style(
    [
        ("question", "bold"),
        ("selected", "fg:#000000 bg:#face0a bold"),
        ("highlighted", "fg:#face0a bold"),
        ("instruction", "fg:#888888"),
        ("separator", "fg:#666666"),
        ("text", ""),
        ("qmark", "fg:#face0a"),
    ]
)

console = Console(highlight=False)
app = typer.Typer(add_completion=False)


def parse_user_input(text):
    """Convert @path/to/image.jpg syntax to standard messages format."""
    image_pattern = r"@([^\s]+\.(?:jpg|jpeg|png|gif|webp))"
    matches = list(re.finditer(image_pattern, text, re.IGNORECASE))

    if not matches:
        # No images, return simple text message
        return [{"role": "user", "content": text}]

    # Build content list with text and images
    content = []
    last_end = 0

    for match in matches:
        # Add text before image
        if match.start() > last_end:
            text_part = text[last_end : match.start()].strip()
            if text_part:
                content.append({"type": "text", "text": text_part})

        # Add image
        image_path = match.group(1)
        if os.path.exists(image_path):
            content.append({"type": "image", "image": image_path})
            console.print(f"✓ Found image: {image_path}", style="green")
        else:
            console.print(f"Warning: Image not found: {image_path}", style="yellow")

        last_end = match.end()

    # Add remaining text
    if last_end < len(text):
        remaining_text = text[last_end:].strip()
        if remaining_text:
            content.append({"type": "text", "text": remaining_text})

    return [{"role": "user", "content": content}]


def generate_local_response(
    messages, model, processor, model_generation, max_tokens=2048, stream=False
):
    """Generate response using local model."""
    # Use processor directly - it now handles both message formats
    inputs = processor(messages)

    # Move inputs to same device as model
    device = next(model.parameters()).device
    inputs["input_ids"] = inputs["input_ids"].to(device)
    if inputs["pixels"] is not None:
        inputs["pixels"] = inputs["pixels"].to(device)
    if inputs["d_image"] is not None:
        inputs["d_image"] = inputs["d_image"].to(device)

    # Default stop tokens for Qwen models
    stop_tokens = [151645, 151644, 151643]  # <|im_end|>, <|im_start|>, <|endoftext|>

    # Generate
    with torch.no_grad():
        if model_generation == "Qwen2.5-VL":
            if inputs["pixels"] is not None:
                # Vision model with images
                generation = model.generate(
                    input_ids=inputs["input_ids"],
                    pixels=inputs["pixels"],
                    d_image=inputs["d_image"],
                    max_new_tokens=max_tokens,
                    stop_tokens=stop_tokens,
                    stream=stream,
                )
            else:
                # Vision model, text-only
                generation = model.generate(
                    input_ids=inputs["input_ids"],
                    pixels=None,
                    d_image=None,
                    max_new_tokens=max_tokens,
                    stop_tokens=stop_tokens,
                    stream=stream,
                )
        else:
            # Text-only model
            generation = model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=max_tokens,
                stop_tokens=stop_tokens,
                stream=stream,
            )

    if stream:
        # Streaming: yield decoded tokens one by one
        for token_id in generation:
            token_text = processor.tokenizer.decode([token_id])
            yield token_text
    else:
        # Non-streaming: decode full response
        input_length = inputs["input_ids"].shape[1]
        response_ids = generation[:, input_length:]
        response = processor.tokenizer.decode(response_ids[0].tolist())
        return response


@app.command()
def main():

    try:
        # Clear the terminal
        os.system("cls" if os.name == "nt" else "clear")

        # Show logo and help message
        yellow_logo = Text(ASCII_LOGO, style="#face0a")
        console.print(yellow_logo)
        console.print(STARTING_HELP_TEXT)

        # Select model generation e.g. Qwen2, Qwen2.5, Qwen2.5-VL, Qwen3, etc.
        selected_model_generation = questionary.select(
            message="Select model",
            choices=[Choice(generation, generation) for generation in ALL_MODELS],
            pointer=">",
            qmark="",
            style=STYLE,
        ).ask()

        if not selected_model_generation:
            return

        # Select model variant e.g. Qwen2-0.5B-Instruct, Qwen2.5-1.5B-Instruct, etc.
        selected_model_variant = questionary.select(
            message="Select model variant",
            choices=[
                Choice(variant, variant)
                for variant in ALL_MODELS[selected_model_generation]
            ],
            pointer=">",
            qmark="",
            style=STYLE,
        ).ask()

        if not selected_model_variant:
            return

        hf_repo_id = ALL_MODELS[selected_model_generation][selected_model_variant]

        console.print(f"\nLoading model: [bold]{hf_repo_id}[/bold]")

        model_class = REPO_ID_TO_MODEL_CLASS.get(hf_repo_id)
        if not model_class:
            console.print("Invalid model variant")
            return

        try:
            model = model_class.from_pretrained(hf_repo_id)
            console.print("Model loaded successfully!")
        except Exception as e:
            console.print(f"Failed to load model: {e}")
            return

        # Create processor with vision config if it's a vision model
        if selected_model_generation == "Qwen2.5-VL":
            from model.vision import VisionConfig

            vision_config = VisionConfig(
                n_embed=model.config.vision_config.n_embed,
                n_layer=model.config.vision_config.n_layer,
                n_heads=model.config.vision_config.n_heads,
                output_n_embed=model.config.n_embed,
                in_channels=model.config.vision_config.in_channels,
                spatial_merge_size=model.config.vision_config.spatial_merge_size,
                spatial_patch_size=model.config.vision_config.spatial_patch_size,
                temporal_patch_size=model.config.vision_config.temporal_patch_size,
                intermediate_size=getattr(
                    model.config.vision_config, "intermediate_size", None
                ),
                hidden_act=getattr(
                    model.config.vision_config, "hidden_act", "quick_gelu"
                ),
            )
            processor = Processor(repo_id=hf_repo_id, vision_config=vision_config)
        else:
            processor = Processor(repo_id=hf_repo_id)

        if not model or not processor:
            console.print("Failed to initialize processor. Exiting...", style="red")
            return

        # Start the interactive chat loop
        messages = []
        while True:
            # Get user input
            user_input = input("\nUSER: ").strip()

            # Handle special commands
            if user_input == "/exit":
                console.print("Goodbye!")
                break
            elif user_input == "/help":
                console.print(HELP_TEXT)
                continue
            elif not user_input:
                continue

            # Parse user input to messages format
            current_messages = parse_user_input(user_input)
            messages.extend(current_messages)

            try:
                # Generate response using local model with streaming
                print("ASSISTANT: ", end="", flush=True)

                response_tokens = []
                for token in generate_local_response(
                    current_messages,
                    model,
                    processor,
                    selected_model_generation,
                    stream=True,
                ):
                    print(token, end="", flush=True)
                    response_tokens.append(token)

                # Complete the response
                print()  # New line after streaming
                response = "".join(response_tokens).strip()

                # Add assistant's response to conversation
                messages.append({"role": "assistant", "content": response})

            except Exception as e:
                console.print(f"Error generating response: {e}", style="red")
                # Remove the failed user message
                if messages and messages[-1]["role"] == "user":
                    messages.pop()

    except KeyboardInterrupt:
        console.print("\nGoodbye!")
    except Exception as e:
        console.print(f"Error: {e}", style="red")
        raise


if __name__ == "__main__":
    app()
