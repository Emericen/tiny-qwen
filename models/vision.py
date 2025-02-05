import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.config import VisionConfig


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class PatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.spatial_patch_size = config.spatial_patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.n_embed

        kernel_size = [
            config.temporal_patch_size,
            config.spatial_patch_size,
            config.spatial_patch_size,
        ]
        self.proj = nn.Conv3d(
            config.in_channels,
            config.n_embed,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.spatial_patch_size,
            self.spatial_patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )
        return hidden_states


class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class VisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.n_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _apply_rotary_pos_emb_vision(
        tensor: torch.Tensor, freqs: torch.Tensor
    ) -> torch.Tensor:
        orig_dtype = tensor.dtype
        tensor = tensor.float()
        cos = freqs.cos()
        sin = freqs.sin()
        cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
        sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
        output = (tensor * cos) + (VisionAttention._rotate_half(tensor) * sin)
        output = output.to(orig_dtype)
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
        rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.n_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        q = self._apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = self._apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.full(
            [1, seq_length, seq_length],
            torch.finfo(q.dtype).min,
            device=q.device,
            dtype=q.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[
                ...,
                cu_seqlens[i - 1] : cu_seqlens[i],
                cu_seqlens[i - 1] : cu_seqlens[i],
            ] = 0

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        # attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class VisionMlp(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self._quick_gelu(self.fc1(x)))

    @staticmethod
    def _quick_gelu(x):
        return x * torch.sigmoid(1.702 * x)


class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embed, eps=1e-6)
        self.ln_1 = nn.LayerNorm(config.n_embed, eps=1e-6)
        self.attn = VisionAttention(config.n_embed, config.n_heads)
        self.norm2 = nn.LayerNorm(config.n_embed, eps=1e-6)
        self.mlp = VisionMlp(config.n_embed)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2VLVisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = PatchEmbed(config=config)
        self.blocks = nn.ModuleList(
            [Qwen2VLVisionBlock(config) for _ in range(config.n_layer)]
        )
        self.merger = PatchMerger(
            dim=config.output_n_embed,
            context_dim=config.n_embed,
            spatial_merge_size=config.spatial_merge_size,
        )
        head_dim = config.n_embed // config.n_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.spatial_merge_size = config.spatial_merge_size
        self.in_channels = config.in_channels

    def rot_pos_emb(self, d_image: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        sms = self.spatial_merge_size

        for t, h, w in d_image:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.view(h // sms, sms, w // sms, sms).transpose(1, 2)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.view(h // sms, sms, w // sms, sms).transpose(1, 2)
            wpos_ids = wpos_ids.flatten()

            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = d_image[:, 1:].max()

        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(self, pixels: torch.Tensor, d_image: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(pixels)
        rotary_pos_emb = self.rot_pos_emb(d_image)
        cu_seqlens = torch.repeat_interleave(
            d_image[:, 1] * d_image[:, 2], d_image[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
            )

        return self.merger(hidden_states)
