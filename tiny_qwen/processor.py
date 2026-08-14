import torch
import urllib.request
import numpy as np
import json
from pathlib import Path
from PIL import Image
from io import BytesIO
from typing import List, Tuple, Optional
from tokenizers import Tokenizer

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
        image_token_id: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_token_id = image_token_id

    @classmethod
    def from_pretrained(cls, weights_path):
        """Load from a local checkpoint directory (tokenizer.json + configs)."""
        path = Path(weights_path)
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))

        min_pixels, max_pixels = 65536, 16777216
        preprocessor = path / "preprocessor_config.json"
        if preprocessor.exists():
            size = json.loads(preprocessor.read_text()).get("size", {})
            min_pixels = size.get("shortest_edge", min_pixels)
            max_pixels = size.get("longest_edge", max_pixels)

        image_token_id = None
        config = path / "config.json"
        if config.exists():
            image_token_id = json.loads(config.read_text()).get("image_token_id")

        return cls(tokenizer, min_pixels=min_pixels, max_pixels=max_pixels,
                   image_token_id=image_token_id)

    @property
    def stop_tokens(self):
        """Token ids that end a generation turn, resolved from the tokenizer."""
        names = ("<|im_end|>", "<|im_start|>", "<|endoftext|>")
        ids = (self.tokenizer.token_to_id(n) for n in names)
        return [i for i in ids if i is not None]

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
            "position_ids": self.get_position_ids(input_ids, d_image),
        }
        if device is not None:
            for key, value in output.items():
                if value is not None:
                    output[key] = value.to(device)
        return output

    # mRoPE position ids. Text tokens advance all three sections together;
    # each image/video block advances (time, height, width) independently so
    # a patch's position encodes where it sits in the frame.
    def get_position_ids(
        self, input_ids: torch.Tensor, d_image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T = input_ids.shape

        # text-only: sequential position ids repeated over the 3 sections
        if d_image is None:
            position_ids = torch.arange(T, dtype=torch.long)
            return position_ids.unsqueeze(0).expand(3, B, -1)

        position_ids = torch.zeros(3, B, T, dtype=torch.long)
        for batch_idx in range(B):
            seq = input_ids[batch_idx]
            text_idx, image_idx, seq_idx = 0, 0, 0
            while seq_idx < T:
                if seq[seq_idx].item() == self.image_token_id:
                    text_idx, image_idx, seq_idx = self._emit_image_block(
                        position_ids=position_ids,
                        batch_idx=batch_idx,
                        seq_idx=seq_idx,
                        text_idx=text_idx,
                        image_idx=image_idx,
                        d_image=d_image,
                    )
                else:
                    position_ids[:, batch_idx, seq_idx] = text_idx
                    text_idx, seq_idx = text_idx + 1, seq_idx + 1

        return position_ids

    def _emit_image_block(
        self,
        position_ids: torch.Tensor,
        batch_idx: int,
        seq_idx: int,
        text_idx: int,
        image_idx: int,
        d_image: torch.Tensor,
    ) -> Tuple[int, int, int]:
        t_img, h_img, w_img = d_image[image_idx]
        t_img = int(t_img.item())
        h_img = int((h_img // SPATIAL_MERGE_SIZE).item())
        w_img = int((w_img // SPATIAL_MERGE_SIZE).item())

        image_token_count = h_img * w_img
        video_token_count = t_img * image_token_count
        for offset in range(video_token_count):
            target_idx = seq_idx + offset
            remaining = offset % image_token_count
            h_pos = remaining // w_img
            w_pos = remaining % w_img

            position_ids[:, batch_idx, target_idx] = text_idx
            position_ids[1, batch_idx, target_idx] = text_idx + h_pos
            position_ids[2, batch_idx, target_idx] = text_idx + w_pos

        return text_idx + 1, image_idx + 1, seq_idx + video_token_count

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
        """Image -> a sequence of patch vectors (Qwen2-VL dynamic resolution).

        1. resize so height and width divide patch*merge (16*2 = 32) pixels
        2. normalize to zero mean, unit-ish scale
        3. duplicate the still image into 2 identical frames, so images and
           2-frame videos share one code path
        4. cut into 16x16-pixel patches and order them so the 4 patches of
           each 2x2 neighborhood are adjacent — the vision encoder later
           merges every consecutive 4 patches into one token
        """
        # 1. resize (convert first: palettized/RGBA images become plain RGB)
        image = image.convert("RGB")
        height, width = np.array(image).shape[:2]
        resized_height, resized_width = self._resize_image(height, width, num_frames=1)
        frame = np.array(
            image.resize((resized_width, resized_height), resample=Image.BICUBIC),
            dtype=np.float32,
        )

        # 2. normalize
        frame = (frame / 255.0 - IMAGE_MEAN) / IMAGE_STD

        # 3. channels-first, then duplicate to TEMPORAL_PATCH_SIZE frames
        frames = np.tile(
            np.transpose(frame, (2, 0, 1))[np.newaxis], (TEMPORAL_PATCH_SIZE, 1, 1, 1)
        )

        # 4. patchify. Axis legend for the reshape:
        #    (t, frame, C, h_block, h_in_block, px_row, w_block, w_in_block, px_col)
        #    where a "block" is a 2x2 group of patches and px are the 16x16
        #    pixels inside one patch.
        n_frames, channels = frames.shape[:2]
        grid_t = n_frames // TEMPORAL_PATCH_SIZE
        grid_h = resized_height // SPATIAL_PATCH_SIZE
        grid_w = resized_width // SPATIAL_PATCH_SIZE
        patches = frames.reshape(
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
        #    reorder to (t, h_block, w_block, h_in_block, w_in_block, C, frame, px_row, px_col):
        #    sequence = raster over 2x2 blocks, then the 4 patches within each;
        #    feature  = channel * 2 frames * 16 * 16 pixels, flattened
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channels * TEMPORAL_PATCH_SIZE * SPATIAL_PATCH_SIZE * SPATIAL_PATCH_SIZE,
        )

        return patches.astype(np.float32), grid_t, grid_h, grid_w

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
