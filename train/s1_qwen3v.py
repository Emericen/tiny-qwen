import torch
import torch.nn as nn
from typing import Optional

from PIL import Image
from model.processor import Processor
from model.vision import VisionConfig, Qwen2VLVisionEncoder
from model.qwen3 import Qwen3Config, Qwen3Dense

from data.llava import LLaVAPretrainDataset, LLaVAInstructDataset


class Qwen3V(nn.Module):
    def __init__(self, vision_config: VisionConfig, lm_config: Qwen3Config):
        super().__init__()
        self.config = lm_config
        self.vision_config = vision_config

        # frozen vision encoder taken from qwen2.5-vl.
        self.visual = Qwen2VLVisionEncoder(vision_config)

        # the only trainable part of this model.
        self.projection = nn.Linear(vision_config.output_n_embed, lm_config.n_embed)

        # frozen qwen3 model.
        self.model = Qwen3Dense(lm_config)
        self.lm_head = None
        if not lm_config.tie_word_embeddings:
            self.lm_head = nn.Linear(
                lm_config.n_embed, lm_config.vocab_size, bias=False
            )

        # Constants for vision tokens
        self.image_pad_token_id = 151655

    def _get_position_ids(
        self, input_ids: torch.Tensor, d_image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Copy directly from Qwen2VL._get_position_ids()
        B, T = input_ids.shape
        device = input_ids.device
        all_pos_ids = torch.zeros(B, 3, T, dtype=torch.long, device=device)

        for batch_idx in range(B):
            seq = input_ids[batch_idx]
            seq_idx = 0
            image_idx = 0
            pos_chunks = []
            position_id = 0

            while seq_idx < T:
                token_id = seq[seq_idx].item()
                if token_id == self.image_pad_token_id:
                    t, h, w = d_image[image_idx]
                    h = h // self.vision_config.spatial_merge_size
                    w = w // self.vision_config.spatial_merge_size

                    t_idx = torch.arange(t).view(t, 1).expand(t, h * w).flatten()
                    h_idx = torch.arange(h).view(1, h, 1).expand(t, h, w).flatten()
                    w_idx = torch.arange(w).view(1, 1, w).expand(t, h, w).flatten()

                    pos_vision = torch.stack([t_idx, h_idx, w_idx]) + position_id
                    pos_chunks.append(pos_vision)
                    position_id = pos_vision.max().item() + 1
                    seq_idx += t * h * w
                    image_idx += 1
                else:
                    pos_text = torch.tensor([position_id])
                    pos_text = pos_text.unsqueeze(0).expand(3, 1)
                    pos_chunks.append(pos_text)
                    position_id += 1
                    seq_idx += 1

            pos_ids_example = torch.cat(pos_chunks, dim=1).to(device)
            all_pos_ids = pos_ids_example.unsqueeze(1).expand(-1, B, -1)

        return all_pos_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor] = None,
        d_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_embeds = self.model.model.embed_tokens(input_ids)

        if pixels is not None:
            # encode images through the vision encoder.
            image_embeds = self.visual(pixels=pixels, d_image=d_image)
            # Project vision embeddings to match text embedding dimension
            image_embeds = self.projection(image_embeds)
            # create a mask for the image tokens of shape (B, T)
            image_mask = input_ids == self.image_pad_token_id
            # expand the mask along embedding dimension to shape (B, T, C)
            image_mask = image_mask.unsqueeze(-1).expand_as(input_embeds)
            # replace image pad token embeddings with actual image embeddings
            input_embeds = input_embeds.masked_scatter(image_mask, image_embeds)

        # Use 3D position ids for multimodal support
        position_ids = self._get_position_ids(input_ids, d_image)
        x = self.model.model(x=input_embeds, position_ids=position_ids)

        if self.lm_head is None:
            logits = torch.matmul(x, self.model.model.embed_tokens.weight.T)
        else:
            logits = self.lm_head(x)
        return logits


if __name__ == "__main__":
    # Qwen2.5 VL 7B's vision config:
    qwen2_5_VL_7B_vision_config_dict = {
        "n_embed": 1280,
        "n_layer": 32,
        "n_heads": 16,
        "output_n_embed": 3584,
        "in_channels": 3,
        "spatial_merge_size": 2,
        "spatial_patch_size": 14,
        "temporal_patch_size": 2,
        "intermediate_size": 3420,
        "hidden_act": "silu",
    }
    qwen2_5_VL_7B_vision_config = VisionConfig(**qwen2_5_VL_7B_vision_config_dict)

    # Qwen3 1.7B's lm config:
    qwen3_1_7B_lm_config_dict = {
        "n_embed": 2048,
        "n_heads": 16,
        "n_kv_heads": 8,
        "n_layer": 28,
        "n_mlp": 6144,
        "rope_theta": 1000000,
        "rms_norm_eps": 1e-06,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
        "head_dim": 128,
        "num_experts": None,
        "num_experts_per_tok": None,
        "moe_intermediate_size": None,
    }
    qwen3_1_7B_lm_config = Qwen3Config(**qwen3_1_7B_lm_config_dict)

    processor = Processor(
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        vision_config=qwen2_5_VL_7B_vision_config,
    )
    model = Qwen3V(
        vision_config=qwen2_5_VL_7B_vision_config,
        lm_config=qwen3_1_7B_lm_config,
    )
    model = model.cuda()  # Move model to CUDA


    pretrain_dataset = LLaVAPretrainDataset(
        processor=processor,
        data_dir="data/llava/pretrain",
        max_length=1024,
    )
    # instruct_dataset = LLaVAInstructDataset(
    #     processor=processor,
    #     data_dir="data/llava/instruct",
    #     max_length=1024,
    # )

    messages = pretrain_dataset[0]["messages"]

    inputs = processor(messages=messages, device="cuda")
    print("Input shapes:")
    print(f"input_ids: {inputs['input_ids'].shape}")
    print(f"pixels: {inputs['pixels']}")  # Should be None for text-only
    print(f"d_image: {inputs['d_image']}")  # Should be None for text-only

    # Test forward pass
    model.eval()
    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"],
            pixels=inputs["pixels"],
            d_image=inputs["d_image"],
        )
        print(f"Output logits shape: {logits.shape}")
