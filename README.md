<p align="left">
    English | <a href="README_CN.md">中文</a>
</p>

<p align="center">
    <img src="assets/banner.png" alt="Tiny Qwen" width="90%">
</p>

## ✨ Tiny Qwen

A minimal re-implementation of Qwen 3.8 with PyTorch. Comes with a single-file, all-purpose agentic harness and int4 quantization support.

If you find Hugging Face code hard to read, you're in the right place.

Join my [Discord channel](https://discord.gg/sBNnqP9gaY) for more discussion!

## 🎇 Quick Start

```bash
# Install dependencies
pip install uv
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Launch the harness with Qwen 3.8 27B
python run.py
```

Or pick any other model or quantization with launch parameters:

```bash
python run.py Qwen/Qwen3.5-4B            # any HF repo id — fetched on demand
python run.py Qwen/Qwen3.5-27B --bits 4  # download, quantize to int4, then run
python run.py weights/Qwen3.5-27B-int4   # any local dir, e.g. a quantized one
```

Or bypass the local model entirely and point the same agentic harness at any OpenAI-compatible endpoint:

```bash
python run.py --url https://api.fireworks.ai/inference/v1/chat/completions \
              --api-key $YOUR_KEY --model accounts/fireworks/models/kimi-k3
```

## 📚 Use it as a library

```python
from tiny_qwen import Model, Processor

model = Model.from_pretrained("weights/Qwen3.5-4B")
processor = Processor.from_pretrained("weights/Qwen3.5-4B")

messages = [
    {"role": "user", "content": [
        {"type": "text", "text": "Describe this image in one sentence."},
        {"type": "image", "image": "photos/cat.jpg"},
    ]},
]
inputs = processor(messages, add_generation_prompt=True, device="mps")

for token_id in model.generate_stream(
    **inputs, max_new_tokens=256, stop_tokens=processor.stop_tokens
):
    print(processor.tokenizer.decode([token_id]), end="", flush=True)
```

## Older versions

`main` runs the whole Qwen 3.5 / 3.6 / 3.8 architecture generation. Older architectures live in branches: [Qwen 3 VL](https://github.com/Emericen/tiny-qwen/tree/legacy/qwen3_vl), [Qwen 3 & Qwen 2.5 VL](https://github.com/Emericen/tiny-qwen/tree/legacy/qwen2_5).

## License

MIT