import json
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
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
    image_token_id: Optional[int] = None  # from config.json; None for text-only models

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
    mrope_section: Optional[List[int]] = None


class LayerCache:
    """One layer's inference state. Attention layers use k/v (an append-only
    ledger that grows with the sequence); GDN layers use S/conv (fixed-size
    fast-weight memory and conv window). The model's whole cache is a plain
    list of these, one per layer; each mixer creates its own via init_cache().
    """

    def __init__(self):
        self.k = None
        self.v = None
        self.S = None
        self.conv = None

    def clone(self):
        """Deep-copy, so a shared prefix (e.g. an encoded video) can be forked
        per question without re-encoding it or contaminating other forks."""
        new = LayerCache()
        for name in ("k", "v", "S", "conv"):
            t = getattr(self, name)
            setattr(new, name, t.clone() if t is not None else None)
        return new


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Qwen3.5 uses partial rotary: only head_dim * partial_rotary_factor dims are rotated.
        dim = int(config.d_head * config.partial_rotary_factor)
        t = config.rope_theta
        r = torch.arange(0, dim, 2)
        self.register_buffer("inv_freq", 1.0 / (t ** (r / dim)).float(), persistent=False)

        self.mrope_section = config.mrope_section or [11, 11, 10]

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

    def init_cache(self, batch_size):
        """An empty ledger: zero-length k/v that the forward appends to."""
        w = self.k_norm.weight  # never quantized, so always has dtype/device
        cache = LayerCache()
        cache.k = w.new_zeros(batch_size, self.n_kv_heads, 0, self.d_head)
        cache.v = w.new_zeros(batch_size, self.n_kv_heads, 0, self.d_head)
        return cache

    def forward(self, x, cos, sin, cache):
        B, T, _ = x.size()

        # split q_proj output into query and gate
        qg = self.q_proj(x).view(B, T, self.n_heads, self.d_head * 2)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(B, T, self.n_heads * self.d_head)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        q, k = self._apply_partial_rotary_pos_emb(q, k, cos, sin)

        past = cache.k.shape[2]
        k = torch.cat([cache.k, k], dim=2)
        v = torch.cat([cache.v, v], dim=2)
        cache.k, cache.v = k, v

        if self.n_kv_heads < self.n_heads:
            num_repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(num_repeat, dim=1)
            v = v.repeat_interleave(num_repeat, dim=1)

        if T > 1 and past > 0:
            # several new tokens on top of a cached prefix: causal mask with offset
            mask = torch.ones(T, past + T, dtype=torch.bool, device=x.device).tril(diagonal=past)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)
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

    def init_cache(self, batch_size):
        """An empty memory: all-zero state S and an all-zero conv window
        (zeros are exactly the causal left-padding the first tokens need)."""
        w = self.conv1d.weight
        cache = LayerCache()
        cache.S = w.new_zeros(batch_size, self.n_v_heads, self.d_k, self.d_v, dtype=torch.float32)
        cache.conv = w.new_zeros(batch_size, w.shape[0], w.shape[2] - 1)
        return cache

    def forward(self, x, cache):
        B, T, _ = x.shape
        H = self.n_v_heads
        r = self.n_v_heads // self.n_k_heads

        # per-token, per-head scalars: write strength in (0,1), log-decay < 0
        beta = self.in_proj_b(x).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(x).float() + self.dt_bias
        )

        # causal depthwise conv over (cached window ++ new tokens), then SiLU
        qkv = self.in_proj_qkv(x).transpose(1, 2)  # (B, conv_dim, T)
        K = self.conv1d.weight.shape[2]
        qkv = torch.cat([cache.conv, qkv], dim=2)  # (B, conv_dim, K-1+T)
        cache.conv = qkv[:, :, -(K - 1):].detach()
        qkv = F.silu(F.conv1d(qkv, self.conv1d.weight, groups=qkv.shape[1]))
        qkv = qkv.transpose(1, 2)  # (B, T, conv_dim)

        # split into heads; each key head serves r value heads
        q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.view(B, T, self.n_k_heads, self.d_k).repeat_interleave(r, dim=2)
        k = k.view(B, T, self.n_k_heads, self.d_k).repeat_interleave(r, dim=2)
        v = v.view(B, T, H, self.d_v)

        # unit-norm queries and keys: bounded writes, attention-style q scale
        q = self._l2norm(q.float()) / self.d_k ** 0.5
        k = self._l2norm(k.float())
        v = v.float()
        beta = beta.float()
        g = g.float()

        # the recurrence: forget, surprise, write, read (float32 — bf16 error
        # compounds over a long sequence; decode is the same loop with T=1)
        S = cache.S
        out = torch.empty(B, T, H, self.d_v, device=x.device, dtype=torch.float32)
        for t in range(T):
            S = S * g[:, t].exp()[:, :, None, None]
            delta = beta[:, t, :, None] * (v[:, t] - torch.einsum("bhkv,bhk->bhv", S, k[:, t]))
            S = S + torch.einsum("bhk,bhv->bhkv", k[:, t], delta)
            out[:, t] = torch.einsum("bhkv,bhk->bhv", S, q[:, t])
        cache.S = S

        # gated output norm + project
        z = self.in_proj_z(x).view(B, T, H, self.d_v)
        y = self.norm(out.to(x.dtype).reshape(-1, self.d_v), z.reshape(-1, self.d_v))
        return self.out_proj(y.view(B, T, -1))

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

    @property
    def mixer(self):
        return self.linear_attn if self.layer_type == "linear_attention" else self.self_attn

    def forward(self, x, cos, sin, cache):
        if self.layer_type == "linear_attention":
            x = x + self.linear_attn(self.input_layernorm(x), cache=cache)
        else:
            x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache=cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Model(nn.Module):
    def __init__(
        self, config: ModelConfig, vision_config: Optional[VisionConfig] = None
    ):
        super().__init__()
        self.config = config
        self.vision_config = vision_config

        self.embed_tokens = nn.Embedding(config.n_vocab, config.n_embed)
        self.rotary_emb = RotaryEmbedding(config)
        self.layers = nn.ModuleList(
            Block(config, layer_idx=i) for i in range(config.n_layer)
        )
        self.norm = GemmaRMSNorm(config.n_embed, eps=config.rms_norm_eps)
        self.lm_head = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.n_embed, config.n_vocab, bias=False)
        self.visual = None
        if vision_config is not None:
            self.visual = VisionEncoder(vision_config)

    def alloc_cache(self, batch_size=1):
        """A fresh, empty inference state: one LayerCache per layer, each
        created by its own mixer (zero memory for GDN, empty ledger for
        attention)."""
        return [block.mixer.init_cache(batch_size) for block in self.layers]

    def _text_position_ids(self, input_ids):
        """Sequential positions repeated over the 3 mRoPE sections. Vision
        inputs need the real thing from Processor.get_position_ids()."""
        B, T = input_ids.shape
        pos = torch.arange(T, dtype=torch.long, device=input_ids.device)
        return pos.unsqueeze(0).expand(3, B, -1)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor] = None,
        d_image: Optional[torch.Tensor] = None,
        cache: Optional[List[LayerCache]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cache is None:
            cache = self.alloc_cache(input_ids.shape[0])  # throwaway
        if position_ids is None:
            if d_image is not None:
                raise ValueError(
                    "vision inputs need position_ids — use the ones returned by the processor"
                )
            position_ids = self._text_position_ids(input_ids)

        x = self.embed_tokens(input_ids)
        if pixels is not None:
            vision_embed = self.visual(pixels=pixels.to(x.dtype), d_image=d_image)
            vision_mask = input_ids == self.config.image_token_id
            if vision_mask.sum().item() != vision_embed.shape[0]:
                raise RuntimeError(
                    "Vision token/feature mismatch: "
                    f"mask_tokens={vision_mask.sum().item()} "
                    f"vision_features={vision_embed.shape[0]} "
                    f"image_token_id={self.config.image_token_id}"
                )
            x[vision_mask] = vision_embed

        cos, sin = self.rotary_emb(x, position_ids)
        for layer, layer_cache in zip(self.layers, cache):
            x = layer(x, cos, sin, cache=layer_cache)
        x = self.norm(x)

        if self.lm_head is None:
            return F.linear(x, self.embed_tokens.weight)
        return self.lm_head(x)

    # ------------------------------------------------------------- loading

    @staticmethod
    def _read_config(model_path: Path):
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
            image_token_id=hf_config.get("image_token_id"),
            rope_theta=llm_config.get("rope_parameters", {}).get("rope_theta")
                       or llm_config.get("rope_theta"),
            rms_norm_eps=llm_config["rms_norm_eps"],
            d_head=llm_config.get("head_dim"),
            n_experts=llm_config.get("num_experts"),
            n_experts_per_token=llm_config.get("num_experts_per_tok"),
            n_moe_mlp=llm_config.get("moe_intermediate_size"),
            n_shared_expert_mlp=llm_config.get("shared_expert_intermediate_size"),
            layer_types=llm_config.get("layer_types"),
            n_linear_k_heads=llm_config.get("linear_num_key_heads"),
            n_linear_v_heads=llm_config.get("linear_num_value_heads"),
            d_linear_k=llm_config.get("linear_key_head_dim"),
            d_linear_v=llm_config.get("linear_value_head_dim"),
            linear_conv_kernel=llm_config.get("linear_conv_kernel_dim", 4),
            partial_rotary_factor=llm_config.get("rope_parameters", {}).get("partial_rotary_factor")
                       or llm_config.get("partial_rotary_factor", 1.0),
            mrope_section=llm_config.get("rope_parameters", {}).get("mrope_section"),
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

        return config, vision_config

    @classmethod
    def _skeleton(cls, config, vision_config):
        """The model with no storage behind it (meta device) — parameters get
        materialized straight from the checkpoint, so peak memory is the
        checkpoint size, never checkpoint + init. The two rotary modules hold
        computed buffers that aren't in checkpoints, so they're rebuilt for
        real, as is the (small) vision encoder."""
        with torch.device("meta"):
            model = cls(config, vision_config)
        model.rotary_emb = RotaryEmbedding(config)
        if vision_config is not None:
            model.visual = VisionEncoder(vision_config)
        return model

    @staticmethod
    def _rename(key):
        """HF checkpoint names -> ours. The checkpoint nests everything under
        model.language_model / model.visual; we don't. MTP weights (unused
        speculative-decoding head) are skipped."""
        if key.startswith("mtp."):
            return None
        for prefix, new in (
            ("model.language_model.", ""),
            ("model.visual.", "visual."),
            ("model.", ""),
        ):
            if key.startswith(prefix):
                return new + key[len(prefix):]
        return key  # lm_head.weight

    @staticmethod
    def _pick_device(device):
        if device is not None:
            return device
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @classmethod
    def from_pretrained(cls, weights_path: str, device: Optional[str] = None):
        """Load from a local directory of HF-format safetensors — or from a
        directory produced by quantize.py (detected by its quant.json)."""
        model_path = Path(weights_path)
        if (model_path / "quant.json").exists():
            return cls.load_quantized(model_path, device)

        device = cls._pick_device(device)
        config, vision_config = cls._read_config(model_path)
        model = cls._skeleton(config, vision_config)

        state = {}
        for shard in sorted(model_path.glob("*.safetensors")):
            with safe_open(shard, framework="pt") as f:
                for key in f.keys():
                    name = cls._rename(key)
                    if name is None:
                        continue
                    tensor = f.get_tensor(key)
                    if tensor.dtype == torch.float32:
                        tensor = tensor.to(torch.bfloat16)
                    state[name] = tensor
        model.load_state_dict(state, strict=False, assign=True)

        missing = [n for n, p in model.named_parameters() if p.is_meta]
        if missing:
            raise RuntimeError(f"checkpoint is missing parameters, e.g. {missing[:4]}")
        return model.to(device).eval()

    @classmethod
    def load_quantized(cls, weights_path: str, device: Optional[str] = None):
        """Load a quantize.py-converted directory: quantized Linears become
        QuantLinear (weights stay small in memory), everything else loads
        as-is."""
        device = cls._pick_device(device)
        model_path = Path(weights_path)
        manifest = json.loads((model_path / "quant.json").read_text())
        bits = manifest["bits"]

        config, vision_config = cls._read_config(model_path)
        model = cls._skeleton(config, vision_config)

        modules = dict(model.named_modules())
        for name in manifest["quantized"]:
            module_name = name.rsplit(".", 1)[0]
            parent_name, child_name = module_name.rsplit(".", 1)
            old = modules[module_name]
            setattr(
                modules[parent_name], child_name,
                QuantLinear(old.out_features, old.in_features, bits),
            )

        state = {}
        for shard in sorted(model_path.glob("*.safetensors")):
            with safe_open(shard, framework="pt") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    if tensor.dtype == torch.float32:
                        tensor = tensor.to(torch.bfloat16)
                    state[key] = tensor
        model.load_state_dict(state, strict=False, assign=True)

        missing = [n for n, p in model.named_parameters() if p.is_meta]
        missing += [n for n, b in model.named_buffers() if b.is_meta]
        if missing:
            raise RuntimeError(f"quantized checkpoint is missing tensors, e.g. {missing[:4]}")
        return model.to(device).eval()

    # ---------------------------------------------------------- generation

    def _generate_core(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor],
        d_image: Optional[torch.Tensor],
        max_new_tokens: int,
        stop_tokens: list,
        position_ids: Optional[torch.Tensor] = None,
    ):
        if stop_tokens is None:
            raise ValueError("stop_tokens is required — use processor.stop_tokens")

        self.eval()
        device = input_ids.device
        B = input_ids.shape[0]

        with torch.no_grad():
            cache = self.alloc_cache(B)
            if position_ids is None:
                position_ids = self._text_position_ids(input_ids)
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
        position_ids: Optional[torch.Tensor] = None,
    ):
        generated_ids = input_ids
        for _, generated_ids in self._generate_core(
            input_ids=input_ids, pixels=pixels, d_image=d_image,
            max_new_tokens=max_new_tokens, stop_tokens=stop_tokens,
            position_ids=position_ids,
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
        position_ids: Optional[torch.Tensor] = None,
    ):
        for token_id, _ in self._generate_core(
            input_ids=input_ids, pixels=pixels, d_image=d_image,
            max_new_tokens=max_new_tokens, stop_tokens=stop_tokens,
            position_ids=position_ids,
        ):
            yield token_id

    @torch.no_grad()
    def encode_prefix(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor] = None,
        d_image: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """Encode a shared prefix (e.g. system prompt + video) once into a
        cache. Reuse it via generate_from_cache() to answer many questions
        about the same prefix without re-encoding it."""
        self.eval()
        cache = self.alloc_cache(input_ids.shape[0])
        if position_ids is None:
            if d_image is not None:
                raise ValueError("vision prefixes need position_ids from the processor")
            position_ids = self._text_position_ids(input_ids)
        self.forward(
            input_ids=input_ids, pixels=pixels, d_image=d_image,
            cache=cache, position_ids=position_ids,
        )
        return cache

    def generate_from_cache(
        self,
        prefix_cache: List[LayerCache],
        prefix_ids: torch.Tensor,
        input_ids: torch.Tensor,
        d_image: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1,
        stop_tokens: list = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """Continue generation from a pre-encoded prefix cache. The prefix is
        forked (cloned) so each call is independent — no re-encoding, no
        cross-question contamination. `position_ids` must cover the FULL
        prefix+question sequence (from the processor when vision is involved);
        text-only defaults to sequential positions. Yields token ids.

        New tokens are fed one at a time (T==1) — correct for both attention
        and the DeltaNet recurrent state; a chunked (T>1) path is future work.
        """
        if stop_tokens is None:
            raise ValueError("stop_tokens is required — use processor.stop_tokens")
        self.eval()
        device = input_ids.device
        B = input_ids.shape[0]

        with torch.no_grad():
            full = torch.cat([prefix_ids, input_ids], dim=1)
            if position_ids is None:
                if d_image is not None:
                    raise ValueError("vision prefixes need position_ids from the processor")
                position_ids = self._text_position_ids(full)
            P = prefix_ids.shape[1]
            new_pos = position_ids[:, :, P:]          # question positions, consistent w/ one-shot
            cache = [layer_cache.clone() for layer_cache in prefix_cache]
            logits = None
            for i in range(input_ids.shape[1]):       # absorb the question into the fork
                ppos = new_pos[:, :, i:i + 1].contiguous()
                logits = self.forward(input_ids=input_ids[:, i:i + 1], cache=cache, position_ids=ppos)
            cur = int(position_ids.max().item()) + 1  # decode start, matching _generate_core
            for _ in range(max_new_tokens):           # decode the answer
                next_token = F.softmax(logits[:, -1, :], dim=-1).argmax(dim=-1, keepdim=True)
                token_id = next_token[0].item()
                yield token_id
                if token_id in stop_tokens:
                    break
                ppos = torch.full((3, B, 1), cur, dtype=torch.long, device=device)
                logits = self.forward(input_ids=next_token, cache=cache, position_ids=ppos)
                cur += 1


# ---------------------------------------------------------------- quantization
# Weight-only block quantization, from scratch. 32 weights share one fp16
# scale (the same block size the GGUF ecosystem uses):
#
#     scale = max|w| / q_max          per block of 32, q_max = 127 or 7
#     q     = round(w / scale)        int8, or int4 packed two per byte
#     w'    = q * scale               dequantized just-in-time in forward
#
# Halves (int8, ~lossless) or quarters (int4, small quality loss) the memory
# a model needs. Speed is not the goal: the transient dequantize adds traffic,
# so this buys FITTING larger models, not faster tokens.

GROUP = 32  # weights per scale


def quantize_int8(w):
    """(out, in) float -> int8 codes (out, in) + fp16 scales (out, in//GROUP)."""
    out_dim, in_dim = w.shape
    blocks = w.float().reshape(out_dim, in_dim // GROUP, GROUP)
    scale = blocks.abs().amax(dim=-1).clamp(min=1e-8) / 127.0
    q = torch.round(blocks / scale[..., None]).to(torch.int8)
    return q.reshape(out_dim, in_dim), scale.to(torch.float16)


def dequantize_int8(q, scale, dtype):
    out_dim, in_dim = q.shape
    blocks = q.reshape(out_dim, -1, GROUP).to(dtype)
    return (blocks * scale.to(dtype)[..., None]).reshape(out_dim, in_dim)


def quantize_int4(w):
    """(out, in) float -> packed uint8 (out, in//2) + fp16 scales (out, in//GROUP).
    Values -7..7 stored +8 as 1..15; two per byte."""
    out_dim, in_dim = w.shape
    blocks = w.float().reshape(out_dim, in_dim // GROUP, GROUP)
    scale = blocks.abs().amax(dim=-1).clamp(min=1e-8) / 7.0
    q = torch.clamp(torch.round(blocks / scale[..., None]), -7, 7) + 8
    q = q.to(torch.uint8).reshape(out_dim, in_dim)
    packed = (q[:, 0::2] << 4) | q[:, 1::2]
    return packed, scale.to(torch.float16)


def dequantize_int4(packed, scale, dtype):
    out_dim = packed.shape[0]
    hi, lo = (packed >> 4), (packed & 0x0F)
    q = torch.stack((hi, lo), dim=2).reshape(out_dim, -1).to(dtype) - 8.0
    blocks = q.reshape(out_dim, -1, GROUP)
    return (blocks * scale.to(dtype)[..., None]).reshape(out_dim, -1)


class QuantLinear(nn.Module):
    """A Linear whose weight lives in quantized blocks, dequantized
    just-in-time in forward and freed after the matmul."""

    def __init__(self, out_features, in_features, bits):
        super().__init__()
        self.bits = bits
        self.out_features = out_features
        self.in_features = in_features
        cols = in_features if bits == 8 else in_features // 2
        dtype = torch.int8 if bits == 8 else torch.uint8
        self.register_buffer("qweight", torch.empty(out_features, cols, dtype=dtype))
        self.register_buffer(
            "scale", torch.empty(out_features, in_features // GROUP, dtype=torch.float16)
        )

    @classmethod
    def from_weight(cls, weight, bits):
        linear = cls(weight.shape[0], weight.shape[1], bits)
        quantize = quantize_int8 if bits == 8 else quantize_int4
        linear.qweight, linear.scale = quantize(weight)
        return linear

    def forward(self, x):
        dequantize = dequantize_int8 if self.bits == 8 else dequantize_int4
        w = dequantize(self.qweight, self.scale, x.dtype)  # transient; freed after matmul
        return F.linear(x, w)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, bits={self.bits}"


def convert_model(model, bits, skip=("lm_head", "visual")):
    """Replace every language-stack nn.Linear with a QuantLinear, in place."""
    replaced = 0
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear) and not any(s in full for s in skip):
                if child.in_features % GROUP != 0:
                    continue
                setattr(parent, child_name, QuantLinear.from_weight(child.weight, bits))
                replaced += 1
    return replaced
