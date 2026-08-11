import torch
import urllib.request
import numpy as np
import json
from pathlib import Path
from PIL import Image
from io import BytesIO
from typing import List, Tuple, Optional
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

# fmt: off
# Constants for message rendering
USER_MESSAGE_TEMPLATE = "<|im_start|>user\n{content}<|im_end|>\n"
ASSISTANT_MESSAGE_TEMPLATE = "<|im_start|>assistant\n{content}{tool_calls}<|im_end|>\n"
TOOL_MESSAGE_TEMPLATE = "<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
SYSTEM_MESSAGE_TEMPLATE = "<|im_start|>system\n{content}<|im_end|>\n"
IMAGE_PAD_TOKEN = "<|image_pad|>"
IMAGE_TEMPLATE = "<|vision_start|>{content}<|vision_end|>"
TOOL_CALL_TEMPLATE = '<tool_call>\n{{"name": "{name}", "arguments": {arguments}}}\n</tool_call>'
TOOL_RESPONSE_TEMPLATE = "<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"

# Constants for image processing
IMAGE_MEAN = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
IMAGE_STD = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
SPATIAL_PATCH_SIZE = 16
SPATIAL_MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
# fmt: on


class Processor:
    def __init__(
        self,
        tokenizer: Tokenizer,
        min_pixels: int = 65536,
        max_pixels: int = 16777216,
    ):
        self.tokenizer = tokenizer
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    @classmethod
    def from_pretrained(cls, repo_id: str):
        tokenizer = Tokenizer.from_pretrained(repo_id)

        # Load preprocessor config to get size parameters
        try:
            config_path = hf_hub_download(repo_id, "preprocessor_config.json")
            with open(config_path, "r") as f:
                config = json.load(f)

            # Extract size parameters
            size = config.get("size", {})
            min_pixels = size.get("shortest_edge", 65536)
            max_pixels = size.get("longest_edge", 16777216)
        except Exception:
            # Fallback to defaults if config not found
            min_pixels = 65536
            max_pixels = 16777216

        return cls(tokenizer, min_pixels=min_pixels, max_pixels=max_pixels)

    # Turn openai harmony style messages into model input tensors.
    def __call__(
        self,
        messages: List[dict],
        add_generation_prompt: bool = False,
        enable_thinking: bool = True,
        device: Optional[torch.device] = None,
    ) -> dict:
        pixels_list = []
        d_image_list = []
        messages_str = ""

        for message in messages:
            role = message["role"]
            content = message.get("content", [])

            tool_calls = message.get("tool_calls", [])
            tool_calls = [self._render_tool_call(tool_call) for tool_call in tool_calls]
            tool_call_str = "".join(tool_calls)

            content_str = "".join(
                [
                    self._render_content(item, pixels_list, d_image_list)
                    for item in content
                ]
            )

            if role == "system":
                messages_str += SYSTEM_MESSAGE_TEMPLATE.format(content=content_str)
            elif role == "user":
                messages_str += USER_MESSAGE_TEMPLATE.format(content=content_str)
            elif role == "assistant":
                messages_str += ASSISTANT_MESSAGE_TEMPLATE.format(
                    content=content_str, tool_calls=tool_call_str
                )
            elif role == "tool":
                messages_str += TOOL_RESPONSE_TEMPLATE.format(content=content_str)
            else:
                raise ValueError(f"Unsupported role: {role}")

        # Add generation prompt if requested
        if add_generation_prompt:
            messages_str += "<|im_start|>assistant\n"
            if enable_thinking:
                messages_str += "<think>\n"
            else:
                messages_str += "<think>\n\n</think>\n\n"

        input_ids = self.tokenizer.encode(messages_str).ids
        input_ids = torch.tensor([input_ids], dtype=torch.long)

        if pixels_list:
            pixels_np = np.concatenate(pixels_list, axis=0)
            pixels = torch.tensor(pixels_np, dtype=torch.float)
            d_image = torch.tensor(d_image_list, dtype=torch.long)
        else:
            pixels = None
            d_image = None

        output = {
            "input_ids": input_ids,
            "pixels": pixels,
            "d_image": d_image,
        }
        if device is not None:
            output["input_ids"] = output["input_ids"].to(device)
            if output["pixels"] is not None:
                output["pixels"] = output["pixels"].to(device)
            if output["d_image"] is not None:
                output["d_image"] = output["d_image"].to(device)
        return output

    def _render_content(
        self, content: dict, pixels_list: list, d_image_list: list
    ) -> str:
        if content["type"] == "text":
            return content["text"]
        elif content["type"] == "image":
            image = None
            if "image" in content:
                image_field = content["image"]
                if isinstance(image_field, Image.Image):
                    image = image_field
                else:
                    image = Image.open(image_field)
            elif "url" in content:
                image = self._fetch_img_through_url(content["url"])
            else:
                raise ValueError(
                    f"Image content must have 'image' or 'url' field, got {content}"
                )

            patches, grid_t, grid_h, grid_w = self._process_image(image)

            pixels_list.append(patches)
            d_image_list.append([grid_t, grid_h, grid_w])

            pad_count = (grid_t * grid_h * grid_w) // (SPATIAL_MERGE_SIZE**2)
            pad_tokens = IMAGE_PAD_TOKEN * pad_count
            return IMAGE_TEMPLATE.format(content=pad_tokens)
        else:
            raise ValueError(f"Unsupported content type: {content['type']}")

    def _render_tool_call(self, tool_call: dict) -> str:
        return TOOL_CALL_TEMPLATE.format(
            name=tool_call["name"], arguments=tool_call["arguments"]
        )

    def _fetch_img_through_url(self, url: str) -> Image.Image:
        # Accepts both local file path and remote URL
        if url.startswith(("http://", "https://")):
            with urllib.request.urlopen(url) as response:
                return Image.open(BytesIO(response.read()))
        else:
            return Image.open(url)

    def _process_image(self, image: Image.Image) -> Tuple[np.ndarray, int, int, int]:
        image_np = np.array(image, dtype=np.float32)
        height, width = image_np.shape[:2]
        resized_height, resized_width = self._resize_image(height, width, num_frames=1)
        image_resized = image.resize(
            (resized_width, resized_height), resample=Image.BICUBIC
        )
        image_np_resized = np.array(image_resized, dtype=np.float32)

        # Normalize
        image_np_resized = image_np_resized / 255.0
        image_np_resized = (image_np_resized - IMAGE_MEAN) / IMAGE_STD

        # Convert to channels-first and add batch dimension
        image_np_resized = np.transpose(image_np_resized, (2, 0, 1))
        image_np_resized = image_np_resized[np.newaxis, ...]

        # Handle temporal dimension
        if image_np_resized.shape[0] == 1:
            image_np_resized = np.tile(image_np_resized, (TEMPORAL_PATCH_SIZE, 1, 1, 1))

        # Extract patches
        batch_size, channels, height, width = image_np_resized.shape
        grid_t = batch_size // TEMPORAL_PATCH_SIZE
        grid_h = resized_height // SPATIAL_PATCH_SIZE
        grid_w = resized_width // SPATIAL_PATCH_SIZE

        patches = image_np_resized.reshape(
            grid_t,
            TEMPORAL_PATCH_SIZE,
            channels,
            grid_h // SPATIAL_MERGE_SIZE,
            SPATIAL_MERGE_SIZE,
            SPATIAL_PATCH_SIZE,
            grid_w // SPATIAL_MERGE_SIZE,
            SPATIAL_MERGE_SIZE,
            SPATIAL_PATCH_SIZE,
        )

        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channels * TEMPORAL_PATCH_SIZE * SPATIAL_PATCH_SIZE * SPATIAL_PATCH_SIZE,
        )

        return flatten_patches.astype(np.float32), grid_t, grid_h, grid_w

    def _resize_image(
        self, height: int, width: int, num_frames: int = 1
    ) -> Tuple[int, int]:
        temporal_factor = TEMPORAL_PATCH_SIZE
        factor = SPATIAL_PATCH_SIZE * SPATIAL_MERGE_SIZE
        if height < factor or width < factor:
            raise ValueError(
                f"height:{height} or width:{width} must be larger than factor:{factor}"
            )
        elif max(height, width) / min(height, width) > 200:
            raise ValueError(
                f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
            )

        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        t_bar = int(np.ceil(num_frames / temporal_factor) * temporal_factor)

        if t_bar * h_bar * w_bar > self.max_pixels:
            beta = np.sqrt((num_frames * height * width) / self.max_pixels)
            h_bar = max(factor, int(np.floor(height / beta / factor) * factor))
            w_bar = max(factor, int(np.floor(width / beta / factor) * factor))
        elif h_bar * w_bar < self.min_pixels:
            # Check 2D area only (without temporal dimension) for min_pixels
            beta = np.sqrt(self.min_pixels / (height * width))
            h_bar = int(np.ceil(height * beta / factor) * factor)
            w_bar = int(np.ceil(width * beta / factor) * factor)

        return h_bar, w_bar
