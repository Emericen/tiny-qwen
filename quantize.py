"""Weight-only quantization, from scratch. The whole scheme is three lines of math:

    scale = max(|W|) / 127            # int8: per output channel
    Wq    = clamp(round(W / scale))   # int4: per group of 128 inputs, packed 2/byte
    y     = x @ (Wq * scale).T        # dequantize just-in-time in forward, then free

Round-to-nearest, symmetric (no zero points). Halves (int8) or quarters (int4) the
memory a model needs. Speed is not the goal: at batch 1 the transient dequantize adds
traffic, so this buys FITTING larger models, not faster tokens.

    python quantize.py Qwen/Qwen3.5-4B --bits 4                # -> .cache/…-int4.pt
    python quantize.py Qwen/Qwen3.5-4B --bits 8 --test         # quantize, then chat once

Loading elsewhere:

    from quantize import load_quantized
    model = load_quantized("path/to/…-int4.pt", device="mps")
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP = 128  # int4 group size along the input dimension


# ---------------------------------------------------------------- the math


def quantize_int8(w):
    """Per-output-channel symmetric int8. w: [out, in] -> (int8 [out, in], scales [out])."""
    scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    q = torch.clamp(torch.round(w / scale[:, None]), -127, 127).to(torch.int8)
    return q, scale.to(torch.float16)

def quantize_int4(w):
    """Per-group symmetric int4, two values packed per byte.
    w: [out, in] -> (uint8 [out, in//2], scales [out, in//GROUP])."""
    out_dim, in_dim = w.shape
    assert in_dim % GROUP == 0, f"in_dim {in_dim} not divisible by group {GROUP}"
    grouped = w.reshape(out_dim, in_dim // GROUP, GROUP)
    scale = grouped.abs().amax(dim=2).clamp(min=1e-8) / 7.0
    q = torch.clamp(torch.round(grouped / scale[:, :, None]), -7, 7) + 8  # store 1..15
    q = q.to(torch.uint8).reshape(out_dim, in_dim)
    packed = (q[:, 0::2] << 4) | q[:, 1::2]
    return packed, scale.to(torch.float16)

def dequantize_int8(q, scale, dtype):
    return q.to(dtype) * scale.to(dtype)[:, None]

def dequantize_int4(packed, scale, dtype):
    out_dim = packed.shape[0]
    hi, lo = (packed >> 4), (packed & 0x0F)
    q = torch.stack((hi, lo), dim=2).reshape(out_dim, -1).to(dtype) - 8.0
    grouped = q.reshape(out_dim, scale.shape[1], GROUP)
    return (grouped * scale.to(dtype)[:, :, None]).reshape(out_dim, -1)


# ---------------------------------------------------------------- the module


class QuantLinear(nn.Module):
    def __init__(self, weight, bias, bits):
        super().__init__()
        self.bits = bits
        self.out_features, self.in_features = weight.shape
        q, scale = (quantize_int8 if bits == 8 else quantize_int4)(weight.float())
        self.register_buffer("qweight", q)
        self.register_buffer("scale", scale)
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None

    def forward(self, x):
        deq = dequantize_int8 if self.bits == 8 else dequantize_int4
        w = deq(self.qweight, self.scale, x.dtype)  # transient; freed after matmul
        return F.linear(x, w, self.bias)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, bits={self.bits}"


def convert_model(model, bits, skip=("lm_head",)):
    """Replace every nn.Linear (except `skip`) with a QuantLinear, in place."""
    replaced = 0
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear) and not any(s in full for s in skip):
                if child.in_features % GROUP != 0 and bits == 4:
                    continue  # leave irregular shapes (rare) in full precision
                setattr(parent, child_name, QuantLinear(child.weight, child.bias, bits))
                replaced += 1
    return replaced


# ---------------------------------------------------------------- save / load


def quantize_checkpoint(repo, bits, device="cpu"):
    from huggingface_hub import snapshot_download
    from model.model import Qwen3_5

    path = snapshot_download(repo_id=repo, cache_dir=".cache")
    model = Qwen3_5.from_pretrained(weights_path=path, device_map={"": device})
    n = convert_model(model, bits)
    out = f".cache/{repo.split('/')[-1]}-int{bits}.pt"
    torch.save({"repo": repo, "bits": bits, "state_dict": model.state_dict()}, out)
    return model, out, n


def load_quantized(pt_path, device="mps"):
    from huggingface_hub import snapshot_download
    from model.model import Qwen3_5

    saved = torch.load(pt_path, map_location="cpu", weights_only=False)
    path = snapshot_download(repo_id=saved["repo"], cache_dir=".cache")
    model = Qwen3_5.from_pretrained(weights_path=path, device_map={"": "cpu"})
    convert_model(model, saved["bits"])
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval()


# ---------------------------------------------------------------- cli


def main():
    parser = argparse.ArgumentParser(description="weight-only RTN quantization")
    parser.add_argument("repo", help="e.g. Qwen/Qwen3.5-4B")
    parser.add_argument("--bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--test", action="store_true", help="generate once after quantizing")
    args = parser.parse_args()

    t0 = time.time()
    model, out, n = quantize_checkpoint(args.repo, args.bits)
    size_gb = sum(
        b.numel() * b.element_size() for b in model.state_dict().values()
    ) / 1e9
    print(f"quantized {n} linears to int{args.bits} in {time.time()-t0:.0f}s")
    print(f"saved {out} ({size_gb:.2f} GB of tensors)")

    if args.test:
        from model.processor import Processor

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device).eval()
        processor = Processor.from_pretrained(args.repo)
        msgs = [{"role": "user", "content": [{"type": "text", "text": "Briefly, what is a lighthouse?"}]}]
        inputs = processor(msgs, add_generation_prompt=True, enable_thinking=False, device=device)
        stops = [t for t in (processor.tokenizer.token_to_id(n) for n in ("<|im_end|>", "<|endoftext|>")) if t is not None]
        t0, ids = time.time(), []
        for tok in model.generate_stream(input_ids=inputs["input_ids"], max_new_tokens=64, stop_tokens=stops):
            ids.append(tok)
        dt = time.time() - t0
        print(f"\n{processor.tokenizer.decode(ids)}\n\n[{len(ids)} tokens, {len(ids)/dt:.2f} tok/s]")


if __name__ == "__main__":
    main()
