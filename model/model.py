import json
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import load_checkpoint_and_dispatch
from .vision import VisionEncoder, VisionConfig


@dataclass
class ModelConfig:
    n_embed: int
    n_heads: int
    n_kv_heads: int
    n_layer: int
    n_mlp: int  # dense MLP intermediate size

    n_vocab: int
    tie_word_embeddings: bool

    rope_theta: float
    rms_norm_eps: float
    image_token_id: int = 151655

    # MoE parameters (Qwen3 VL)
    d_head: Optional[int] = None
    n_experts: Optional[int] = None
    n_experts_per_token: Optional[int] = None
    n_moe_mlp: Optional[int] = None
    n_shared_expert_mlp: Optional[int] = None

    # Linear attention parameters (Qwen3.5)
    layer_types: Optional[List[str]] = None
    n_linear_k_heads: Optional[int] = None
    n_linear_v_heads: Optional[int] = None
    d_linear_k: Optional[int] = None
    d_linear_v: Optional[int] = None
    linear_conv_kernel: int = 4
    partial_rotary_factor: float = 1.0


class KVCache:
    """Cache for standard attention layers - stores K/V tensors."""

    def __init__(self, n_layers):
        self.cache = [None] * n_layers

    def update(self, layer_idx, k, v):
        if self.cache[layer_idx] is not None:
            prev_k, prev_v = self.cache[layer_idx]
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        self.cache[layer_idx] = (k, v)
        return k, v

    @property
    def seq_len(self):
        if self.cache[0] is None:
            return 0
        return self.cache[0][0].shape[2]


class DeltaNetCache:
    """Cache for GatedDeltaNet layers - stores recurrent state S and conv state."""

    def __init__(self, n_layers):
        self.cache = [None] * n_layers
        self.conv_cache = [None] * n_layers

    def get(self, layer_idx):
        return self.cache[layer_idx]

    def update(self, layer_idx, S):
        self.cache[layer_idx] = S

    def get_conv(self, layer_idx):
        return self.conv_cache[layer_idx]

    def update_conv(self, layer_idx, state):
        self.conv_cache[layer_idx] = state


class InferenceCache:
    """Unified cache for both attention types."""

    def __init__(self, config):
        self.kv_cache = KVCache(config.n_layer)
        self.delta_cache = DeltaNetCache(config.n_layer)
        self.config = config

    @staticmethod
    def alloc(config, batch_size, device):
        cache = InferenceCache(config)
        return cache


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_head
        t = config.rope_theta
        r = torch.arange(0, d, 2)
        self.register_buffer("inv_freq", 1.0 / (t ** (r / d)).float(), persistent=False)

        self.mrope_section = [24, 20, 20]

    def forward(self, x, position_ids):
        inv_freq = self.inv_freq.to(dtype=torch.float32, device=x.device)
        inv_freq_expanded = inv_freq[None, None, :, None].expand(
            3, position_ids.shape[1], -1, 1
        )
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)

        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
        return cos, sin

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """[TTT...HHH...WWW] -> [THWTHWTHW...TT]"""
        freqs_t = freqs[0]  # start with temporal dimension
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.n_kv_heads = config.n_kv_heads
        self.n_embed = config.n_embed
        self.partial_rotary_factor = config.partial_rotary_factor

        # q_proj outputs 2x: query + gate
        self.q_proj = nn.Linear(self.n_embed, self.n_heads * self.d_head * 2, bias=False)
        self.k_proj = nn.Linear(self.n_embed, self.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(self.n_embed, self.n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.d_head, self.n_embed, bias=False)

        self.q_norm = GemmaRMSNorm(self.d_head, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.d_head, eps=config.rms_norm_eps)

    def forward(self, x, cos, sin, kv_cache=None, layer_idx=None):
        B, T, _ = x.size()

        # split q_proj output into query and gate
        qg = self.q_proj(x).view(B, T, self.n_heads, self.d_head * 2)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(B, T, self.n_heads * self.d_head)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        q, k = self._apply_partial_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        if self.n_kv_heads < self.n_heads:
            num_repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(num_repeat, dim=1)
            v = v.repeat_interleave(num_repeat, dim=1)

        is_causal = T > 1 and (kv_cache is None or kv_cache.seq_len == T)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)
        y = y * torch.sigmoid(gate)
        y = self.o_proj(y)
        return y

    def _apply_partial_rotary_pos_emb(self, q, k, cos, sin):
        rotary_dim = cos.shape[-1]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

        q_rot = (q_rot * cos) + (self._rotate_half(q_rot) * sin)
        k_rot = (k_rot * cos) + (self._rotate_half(k_rot) * sin)

        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class GatedDeltaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_k_heads = config.n_linear_k_heads
        self.n_v_heads = config.n_linear_v_heads
        self.d_k = config.d_linear_k
        self.d_v = config.d_linear_v
        self.key_dim = self.n_k_heads * self.d_k
        self.value_dim = self.n_v_heads * self.d_v
        conv_kernel = config.linear_conv_kernel

        # Keep naming aligned with HF Qwen3.5 checkpoints.
        self.in_proj_qkv = nn.Linear(
            config.n_embed, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(config.n_embed, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.n_embed, self.n_v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.n_embed, self.n_v_heads, bias=False)
        self.out_proj = nn.Linear(self.value_dim, config.n_embed, bias=False)

        conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            conv_dim, conv_dim, conv_kernel,
            groups=conv_dim, padding=conv_kernel - 1, bias=False,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.n_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.n_v_heads).uniform_(0, 16).log())
        self.norm = RMSNormGated(self.d_v, eps=config.rms_norm_eps)

    def forward(self, x, cache=None, layer_idx=None):
        B, T, _ = x.shape

        if cache is not None and T == 1:
            return self._step(x, cache, layer_idx)

        # project + causal conv1d with SiLU
        qkv = self.in_proj_qkv(x)

        if cache is not None:
            K = self.conv1d.weight.shape[2]
            cache.update_conv(layer_idx, qkv[:, -(K - 1):, :].detach().transpose(1, 2))

        qkv = F.silu(self.conv1d(qkv.transpose(1, 2))[:, :, :T]).transpose(1, 2)

        z = self.in_proj_z(x).view(B, T, self.n_v_heads, self.d_v)
        beta = self.in_proj_b(x).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(x).float() + self.dt_bias
        )

        # split and reshape
        q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.view(B, T, self.n_k_heads, self.d_k)
        k = k.view(B, T, self.n_k_heads, self.d_k)
        v = v.view(B, T, self.n_v_heads, self.d_v)

        # GQA expansion
        if self.n_v_heads > self.n_k_heads:
            r = self.n_v_heads // self.n_k_heads
            q = q.repeat_interleave(r, dim=2)
            k = k.repeat_interleave(r, dim=2)

        # delta rule attention
        y, S_final = self._gated_delta_rule(q, k, v, g, beta)

        # cache final state for inference
        if cache is not None:
            cache.update(layer_idx, S_final)

        # gated output norm + project
        y = self.norm(y.reshape(-1, self.d_v), z.reshape(-1, self.d_v))
        return self.out_proj(y.view(B, T, -1))

    def _step(self, x, cache, layer_idx):
        """Single-token inference step with persistent state."""
        B, T, _ = x.shape
        assert T == 1, "Step mode only supports single token"

        z = self.in_proj_z(x).view(B, T, self.n_v_heads, self.d_v)
        beta = self.in_proj_b(x).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(x).float() + self.dt_bias
        )

        qkv = self.in_proj_qkv(x)
        conv_state = cache.get_conv(layer_idx)  # (B, conv_dim, K-1)
        new_qkv = qkv.transpose(1, 2)           # (B, conv_dim, 1)
        conv_input = torch.cat([conv_state, new_qkv], dim=2)  # (B, conv_dim, K)
        cache.update_conv(layer_idx, conv_input[:, :, 1:])
        qkv = F.silu((self.conv1d.weight.squeeze(1) * conv_input).sum(-1)).unsqueeze(1)

        q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.view(B, T, self.n_k_heads, self.d_k)
        k = k.view(B, T, self.n_k_heads, self.d_k)
        v = v.view(B, T, self.n_v_heads, self.d_v)

        # GQA expansion
        if self.n_v_heads > self.n_k_heads:
            r = self.n_v_heads // self.n_k_heads
            q = q.repeat_interleave(r, dim=2)
            k = k.repeat_interleave(r, dim=2)

        # single-step delta rule with cached state
        y = self._gated_delta_rule_step(q, k, v, g, beta, cache, layer_idx)

        # gated output norm + project
        y = self.norm(y.reshape(-1, self.d_v), z.reshape(-1, self.d_v))
        return self.out_proj(y.view(B, T, -1))

    def _gated_delta_rule_step(self, q, k, v, g, beta, cache, layer_idx):
        """Single step of gated delta rule with persistent state."""
        out_dtype = q.dtype
        q = q.transpose(1, 2).contiguous().float()
        k = k.transpose(1, 2).contiguous().float()
        v = v.transpose(1, 2).contiguous().float()
        beta = beta.transpose(1, 2).contiguous().float()
        g = g.transpose(1, 2).contiguous().float()

        q = self._l2norm(q) / (q.shape[-1] ** 0.5)
        k = self._l2norm(k)

        B, H, T, d_k = k.shape
        d_v = v.shape[-1]

        # Get or initialize state
        S = cache.get(layer_idx)
        if S is None:
            S = torch.zeros(B, H, d_k, d_v, device=v.device, dtype=v.dtype)

        q_t, k_t, v_t = q[:, :, 0], k[:, :, 0], v[:, :, 0]
        g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, 0].unsqueeze(-1)

        S = S * g_t
        delta = (v_t - (S * k_t.unsqueeze(-1)).sum(-2)) * beta_t
        S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out = (S * q_t.unsqueeze(-1)).sum(-2)

        # Update cache
        cache.update(layer_idx, S)

        return out.unsqueeze(2).to(out_dtype)

    def _gated_delta_rule(self, q, k, v, g, beta):
        """Recurrent gated delta rule with L2-normalized Q, K. Returns (output, final_state)."""
        out_dtype = q.dtype
        q, k, v, beta, g = [
            x.transpose(1, 2).contiguous().float() for x in (q, k, v, beta, g)
        ]
        q = self._l2norm(q) / (q.shape[-1] ** 0.5)
        k = self._l2norm(k)

        B, H, T, d_k = k.shape
        d_v = v.shape[-1]
        S = torch.zeros(B, H, d_k, d_v, device=v.device, dtype=v.dtype)
        out = torch.zeros_like(v)

        for t in range(T):
            q_t, k_t, v_t = q[:, :, t], k[:, :, t], v[:, :, t]
            g_t = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, t].unsqueeze(-1)

            S = S * g_t
            delta = (v_t - (S * k_t.unsqueeze(-1)).sum(-2)) * beta_t
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            out[:, :, t] = (S * q_t.unsqueeze(-1)).sum(-2)

        return out.transpose(1, 2).contiguous().to(out_dtype), S

    @staticmethod
    def _l2norm(x, eps=1e-6):
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


class RMSNorm(nn.Module):
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


class GemmaRMSNorm(nn.Module):
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_embed))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return ((1.0 + self.weight.float()) * x).to(input_dtype)


class RMSNormGated(nn.Module):
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embed))
        self.variance_epsilon = eps

    def forward(self, x, gate):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = self.weight * x.to(input_dtype)
        return x * F.silu(gate.to(torch.float32)).to(input_dtype)


class DenseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embed, config.n_mlp, bias=False)
        self.up_proj = nn.Linear(config.n_embed, config.n_mlp, bias=False)
        self.down_proj = nn.Linear(config.n_mlp, config.n_embed, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SharedExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.n_embed = config.n_embed
        self.n_moe_mlp = config.n_moe_mlp

        self.gate_up_proj = nn.Parameter(
            torch.empty(self.n_experts, 2 * self.n_moe_mlp, self.n_embed)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.n_experts, self.n_embed, self.n_moe_mlp)
        )

    def forward(
        self,
        x: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        x_out = torch.zeros_like(x)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                top_k_index, num_classes=self.n_experts
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.n_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            x_curr = x[token_idx]
            gate, up = F.linear(x_curr, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            x_curr = F.silu(gate) * up
            x_curr = F.linear(x_curr, self.down_proj[expert_idx])
            x_curr = x_curr * top_k_weights[token_idx, top_k_pos, None]
            x_out.index_add_(0, token_idx, x_curr.to(x_out.dtype))

        return x_out


class MoEMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embed = config.n_embed
        self.n_moe_mlp = config.n_moe_mlp
        self.n_experts = config.n_experts
        self.top_k = config.n_experts_per_token
        self.shared_expert_dim = getattr(config, "n_shared_expert_mlp", None)
        self.gate = nn.Linear(self.n_embed, self.n_experts, bias=False)
        self.experts = MoEExperts(config)
        self.shared_expert = None
        self.shared_expert_gate = None
        if self.shared_expert_dim:
            self.shared_expert = SharedExpertMLP(
                hidden_size=self.n_embed,
                intermediate_size=self.shared_expert_dim,
            )
            self.shared_expert_gate = nn.Linear(self.n_embed, 1, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        x_flat = x.reshape(-1, self.n_embed)

        router_logits = F.linear(x_flat, self.gate.weight)
        router_logits = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
        topk_weights = topk_weights.to(router_logits.dtype)
        expert_out = self.experts(x_flat, topk_indices, topk_weights)

        if self.shared_expert is not None and self.shared_expert_gate is not None:
            shared_expert_out = self.shared_expert(x_flat)
            shared_expert_out = torch.sigmoid(
                self.shared_expert_gate(x_flat)
            ) * shared_expert_out
            expert_out = expert_out + shared_expert_out

        return expert_out.view(B, T, self.n_embed)


class Block(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        n_embed, eps = config.n_embed, config.rms_norm_eps

        layer_type = "full_attention"
        if config.layer_types is not None:
            layer_type = config.layer_types[layer_idx]

        self.layer_type = layer_type
        self.input_layernorm = GemmaRMSNorm(n_embed=n_embed, eps=eps)
        self.post_attention_layernorm = GemmaRMSNorm(n_embed=n_embed, eps=eps)

        if layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config)
        else:
            self.self_attn = SelfAttention(config)

        self.mlp = MoEMLP(config) if config.n_experts else DenseMLP(config)

    def forward(self, x, cos, sin, cache=None, layer_idx=None):
        if self.layer_type == "linear_attention":
            x = x + self.linear_attn(
                self.input_layernorm(x), cache=cache.delta_cache if cache else None, layer_idx=layer_idx
            )
        else:
            x = x + self.self_attn(
                self.input_layernorm(x), cos, sin,
                kv_cache=cache.kv_cache if cache else None,
                layer_idx=layer_idx,
            )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.n_vocab, config.n_embed)
        self.rotary_emb = RotaryEmbedding(config)

        self.layers = nn.ModuleList(
            Block(config, layer_idx=i) for i in range(config.n_layer)
        )
        self.norm = GemmaRMSNorm(config.n_embed, eps=config.rms_norm_eps)

    def forward(
        self,
        input_embed,
        vision_embed=None,
        vision_mask=None,
        position_ids=None,
        cache=None,
    ):
        if vision_embed is not None and vision_mask is not None:
            input_embed[vision_mask] = vision_embed

        cos, sin = self.rotary_emb(input_embed, position_ids)
        for layer_idx, layer in enumerate(self.layers):
            input_embed = layer(input_embed, cos, sin, cache=cache, layer_idx=layer_idx)

        input_embed = self.norm(input_embed)
        return input_embed


class Qwen3_5(nn.Module):
    def __init__(
        self, config: ModelConfig, vision_config: Optional[VisionConfig] = None
    ):
        super().__init__()
        self.config = config
        self.vision_config = vision_config

        self.model = nn.Module()
        self.model.language_model = Model(config)
        self.lm_head = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.n_embed, config.n_vocab, bias=False)

        if vision_config is not None:
            self.model.visual = VisionEncoder(vision_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor] = None,
        d_image: Optional[torch.Tensor] = None,
        cache: Optional[InferenceCache] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_embeds = self.model.language_model.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = self._get_position_ids(input_ids=input_ids, d_image=d_image)

        if pixels is not None:
            pixels = pixels.to(input_embeds.dtype)
            vision_embed = self.model.visual(pixels=pixels, d_image=d_image)
            image_pad_token = self.config.image_token_id
            vision_mask = input_ids == image_pad_token
            if vision_mask.sum().item() != vision_embed.shape[0]:
                raise RuntimeError(
                    "Vision token/feature mismatch: "
                    f"mask_tokens={vision_mask.sum().item()} "
                    f"vision_features={vision_embed.shape[0]} "
                    f"image_token_id={image_pad_token}"
                )

            output = self.model.language_model(
                input_embed=input_embeds,
                vision_embed=vision_embed,
                vision_mask=vision_mask,
                position_ids=position_ids,
                cache=cache,
            )
        else:
            output = self.model.language_model(
                input_embed=input_embeds, position_ids=position_ids, cache=cache
            )

        logits = (
            output @ self.model.language_model.embed_tokens.weight.T
            if self.lm_head is None
            else self.lm_head(output)
        )
        return logits

    def _get_position_ids(
        self, input_ids: torch.Tensor, d_image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T = input_ids.shape
        image_pad_token = self.config.image_token_id

        # text-only case: sequential position IDs repeated 3 times
        if d_image is None:
            position_ids = torch.arange(T, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(3, B, -1)
            return position_ids

        # text + vision case: 3D position IDs
        position_ids = torch.zeros(3, B, T, dtype=torch.long, device=input_ids.device)
        for batch_idx in range(B):
            seq = input_ids[batch_idx]
            text_idx, image_idx, seq_idx = 0, 0, 0
            while seq_idx < T:
                token_id = seq[seq_idx].item()
                if token_id == image_pad_token:
                    # start of an vision block (image(s))
                    text_idx, image_idx, seq_idx = self._emit_image_block(
                        position_ids=position_ids,
                        batch_idx=batch_idx,
                        seq_idx=seq_idx,
                        text_idx=text_idx,
                        image_idx=image_idx,
                        d_image=d_image,
                    )
                else:
                    # treat as regular text token
                    position_ids[:, batch_idx, seq_idx] = text_idx
                    text_idx, image_idx, seq_idx = text_idx + 1, image_idx, seq_idx + 1

        return position_ids

    def _emit_image_block(
        self,
        position_ids: torch.Tensor,
        batch_idx: int,
        seq_idx: int,
        text_idx: int,
        image_idx: int,
        d_image: torch.Tensor,
        spatial_merge_size: int = 2,
    ) -> Tuple[int, int, int]:
        t_img, h_img, w_img = d_image[image_idx]
        t_img = int(t_img.item())
        h_img = int((h_img // spatial_merge_size).item())
        w_img = int((w_img // spatial_merge_size).item())

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

    @classmethod
    def from_pretrained(cls, weights_path: str, device_map: str = "auto"):
        model_path = Path(weights_path)

        with open(model_path / "config.json", "r") as f:
            hf_config = json.load(f)

        llm_config = hf_config["text_config"]

        n_mlp = llm_config.get("intermediate_size")
        if n_mlp is None:
            n_mlp = llm_config.get("shared_expert_intermediate_size")
        if n_mlp is None:
            n_mlp = llm_config.get("moe_intermediate_size")

        config = ModelConfig(
            n_embed=llm_config["hidden_size"],
            n_heads=llm_config["num_attention_heads"],
            n_kv_heads=llm_config["num_key_value_heads"],
            n_layer=llm_config["num_hidden_layers"],
            n_mlp=n_mlp,
            n_vocab=llm_config["vocab_size"],
            tie_word_embeddings=hf_config["tie_word_embeddings"],
            image_token_id=hf_config.get("image_token_id", 151655),
            rope_theta=llm_config.get("rope_parameters", {}).get("rope_theta")
                       or llm_config.get("rope_theta"),
            rms_norm_eps=llm_config["rms_norm_eps"],
            d_head=llm_config.get("head_dim"),
            n_experts=llm_config.get("num_experts"),
            n_experts_per_token=llm_config.get("num_experts_per_tok"),
            n_moe_mlp=llm_config.get("moe_intermediate_size"),
            n_shared_expert_mlp=llm_config.get("shared_expert_intermediate_size"),
            # Qwen3.5 linear attention params
            layer_types=llm_config.get("layer_types"),
            n_linear_k_heads=llm_config.get("linear_num_key_heads"),
            n_linear_v_heads=llm_config.get("linear_num_value_heads"),
            d_linear_k=llm_config.get("linear_key_head_dim"),
            d_linear_v=llm_config.get("linear_value_head_dim"),
            linear_conv_kernel=llm_config.get("linear_conv_kernel_dim", 4),
            partial_rotary_factor=llm_config.get("partial_rotary_factor", 1.0),
        )

        vision_config = None
        vision_config_data = hf_config.get("vision_config")
        if vision_config_data is not None:
            vision_config = VisionConfig(
                n_embed=vision_config_data["hidden_size"],
                n_layer=vision_config_data["depth"],
                n_heads=vision_config_data["num_heads"],
                n_output_embed=vision_config_data["out_hidden_size"],
                n_mlp=vision_config_data["intermediate_size"],
                num_position_embeddings=vision_config_data["num_position_embeddings"],
                in_channels=vision_config_data["in_channels"],
                temporal_patch_size=vision_config_data["temporal_patch_size"],
                patch_size=vision_config_data["patch_size"],
                spatial_merge_size=vision_config_data["spatial_merge_size"],
            )

        model = cls(config, vision_config=vision_config)
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=str(model_path),
            device_map=device_map,
            no_split_module_classes=["Block", "VisionBlock"],
            dtype=torch.bfloat16,
            skip_keys=["mtp."],
        )

        return model

    def _generate_core(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor],
        d_image: Optional[torch.Tensor],
        max_new_tokens: int,
        stop_tokens: Optional[list],
    ):
        if stop_tokens is None:
            # <|im_end|>, <|im_start|>, <|endoftext|>
            stop_tokens = [248046, 248045, 248044]

        self.eval()
        device = input_ids.device
        B = input_ids.shape[0]

        with torch.no_grad():
            cache = InferenceCache.alloc(self.config, B, device)

            position_ids = self._get_position_ids(input_ids=input_ids, d_image=d_image)
            logits = self.forward(
                input_ids=input_ids, pixels=pixels, d_image=d_image,
                cache=cache, position_ids=position_ids,
            )
            next_pos = position_ids.max().item() + 1
            generated_ids = input_ids

            for _ in range(max_new_tokens):
                last_logits = logits[:, -1, :]
                probs = F.softmax(last_logits, dim=-1)
                next_token = probs.argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                token_id = next_token[0].item()
                yield token_id, generated_ids

                if token_id in stop_tokens:
                    break

                decode_pos = torch.full((3, B, 1), next_pos, dtype=torch.long, device=device)
                logits = self.forward(
                    input_ids=next_token, cache=cache, position_ids=decode_pos,
                )
                next_pos += 1

    def generate(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor] = None,
        d_image: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1,
        stop_tokens: list = None,
    ):
        generated_ids = input_ids

        for _, generated_ids in self._generate_core(
            input_ids=input_ids,
            pixels=pixels,
            d_image=d_image,
            max_new_tokens=max_new_tokens,
            stop_tokens=stop_tokens,
        ):
            pass

        return generated_ids

    def generate_stream(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor] = None,
        d_image: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1,
        stop_tokens: list = None,
    ):
        for token_id, _ in self._generate_core(
            input_ids=input_ids,
            pixels=pixels,
            d_image=d_image,
            max_new_tokens=max_new_tokens,
            stop_tokens=stop_tokens,
        ):
            yield token_id
