import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass
from .vision import Qwen2VLVisionEncoder, VisionConfig


@dataclass
class Qwen2Config:
    n_embed: int
    n_heads: int
    n_kv_heads: int
    n_layer: int
    n_mlp: int
    rope_theta: float
    rms_norm_eps: float
    vocab_size: int
    tie_word_embeddings: bool
    vision_config: Optional[VisionConfig] = None
    head_dim: Optional[int] = None  # Explicit head dimension


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Use explicit head_dim if provided, otherwise calculate
        d = (
            config.head_dim
            if config.head_dim is not None
            else (config.n_embed // config.n_heads)
        )
        t = config.rope_theta
        r = torch.arange(0, d, 2)
        self.inv_freq = 1.0 / (t ** (r / d)).float()

    def forward(self, x, position_ids):
        # Check the dimensionality of position_ids to decide shape
        # shape is typically B x T (2D) for text, or B x 3 x T (3D) for multimodal
        if position_ids.dim() == 3:
            inv_freq = self.inv_freq.unsqueeze(0).unsqueeze(0).to(x.device)
        else:
            inv_freq = self.inv_freq.to(x.device)

        position_ids = position_ids.unsqueeze(-1)
        freqs = position_ids * inv_freq
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
        return cos, sin


class Qwen2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads

        self.n_embed = config.n_embed
        self.n_embed_per_head = config.n_embed // config.n_heads
        self.n_kv_embed = config.n_kv_heads * self.n_embed_per_head

        self.q_proj = nn.Linear(self.n_embed, self.n_embed, bias=True)
        self.k_proj = nn.Linear(self.n_embed, self.n_kv_embed, bias=True)
        self.v_proj = nn.Linear(self.n_embed, self.n_kv_embed, bias=True)
        self.o_proj = nn.Linear(self.n_embed, self.n_embed, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.n_embed_per_head).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.n_embed_per_head).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.n_embed_per_head).transpose(1, 2)

        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        if self.n_kv_heads < self.n_heads:
            num_repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(num_repeat, dim=1)
            v = v.repeat_interleave(num_repeat, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        return y

    @staticmethod
    def _apply_rotary_pos_emb(q, k, cos, sin):
        if cos.dim() == 4:
            # shape [B, 3, T, D] -> multi-modal
            cos = Qwen2Attention._process_rotary_component(cos)
            sin = Qwen2Attention._process_rotary_component(sin)
        else:
            # shape [B, T, D] -> text-only
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        q_embed = (q * cos) + (Qwen2Attention._rotate_half(q) * sin)
        k_embed = (k * cos) + (Qwen2Attention._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _process_rotary_component(x):
        # Split into sections and select appropriate indices
        sections = x.split([16, 24, 24, 16, 24, 24], dim=-1)
        processed = [m[i % 3] for i, m in enumerate(sections)]
        # Combine and add dimension
        return torch.cat(processed, dim=-1).unsqueeze(1)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class Qwen2RMSNorm(nn.Module):
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embed))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embed, config.n_mlp, bias=False)
        self.up_proj = nn.Linear(config.n_embed, config.n_mlp, bias=False)
        self.down_proj = nn.Linear(config.n_mlp, config.n_embed, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embed, eps = config.n_embed, config.rms_norm_eps
        self.input_layernorm = Qwen2RMSNorm(n_embed=n_embed, eps=eps)
        self.self_attn = Qwen2Attention(config)
        self.post_attention_layernorm = Qwen2RMSNorm(n_embed=n_embed, eps=eps)
        self.mlp = Qwen2MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embed)
        self.rotary_emb = Qwen2RotaryEmbedding(config)
        self.layers = nn.ModuleList(Qwen2Block(config) for _ in range(config.n_layer))
        self.norm = Qwen2RMSNorm(config.n_embed, eps=config.rms_norm_eps)

    def forward(self, x, position_ids):
        cos, sin = self.rotary_emb(x, position_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return x


class Qwen2VL(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.visual = Qwen2VLVisionEncoder(config.vision_config)
        self.model = Qwen2Model(config)
        self.lm_head = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        self.vision_start_token_id = 151652
        self.image_pad_token_id = 151655
        self.video_pad_token_id = -1  # placeholder

    def _get_position_ids(
        self,
        input_ids: torch.LongTensor,
        d_image: Optional[torch.LongTensor] = None,
    ) -> torch.LongTensor:
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
                    h = h // self.config.vision_config.spatial_merge_size
                    w = w // self.config.vision_config.spatial_merge_size

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
                    pos_text = pos_text.unsqueeze(0).expand(3, 1)  # shape (3,1)
                    pos_chunks.append(pos_text)

                    position_id += 1
                    seq_idx += 1

            # Concatenate all chunks for this example => shape [3, seq_len]
            pos_ids_example = torch.cat(pos_chunks, dim=1).to(device)
            all_pos_ids = pos_ids_example.unsqueeze(1).expand(-1, B, -1)

        return all_pos_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor] = None,
        d_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_embeds = self.model.embed_tokens(input_ids)

        if pixels is not None:
            # encode images through the vision encoder.
            image_embeds = self.visual(pixels=pixels, d_image=d_image)
            # create a mask for the image tokens of shape (B, T)
            image_mask = input_ids == self.image_pad_token_id
            # expand the mask along embedding dimension to shape (B, T, C)
            image_mask = image_mask.unsqueeze(-1).expand_as(input_embeds)
            # replace image pad token embeddings with actual image embeddings
            input_embeds = input_embeds.masked_scatter(image_mask, image_embeds)

        position_ids = self._get_position_ids(input_ids=input_ids, d_image=d_image)
        x = self.model(x=input_embeds, position_ids=position_ids)

        if self.lm_head is None:
            logits = torch.matmul(x, self.model.embed_tokens.weight.T)
        else:
            logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixels: torch.Tensor,
        d_image: torch.Tensor,
        max_new_tokens: int = 1,
        stop_tokens: list = None,
        stream: bool = False,
    ):
        if stop_tokens is None:
            stop_tokens = [
                151645,
                151644,
                151643,
            ]  # <|im_end|>, <|im_start|>, <|endoftext|>

        for _ in range(max_new_tokens):
            logits = self.forward(input_ids=input_ids, pixels=pixels, d_image=d_image)
            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            next_token = probs.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # If streaming, yield the new token
            if stream:
                yield next_token.item()

            # Check if we hit a stop token
            if next_token.item() in stop_tokens:
                break

        # If not streaming, return the full input_ids
        if not stream:
            return input_ids

    @classmethod
    def get_config_class(cls):
        return Qwen2Config

    @classmethod
    def from_pretrained(cls, repo_id: str, device_map: str = "auto"):
        from .util import load_pretrained_model

        return load_pretrained_model(cls, repo_id, device_map=device_map)
