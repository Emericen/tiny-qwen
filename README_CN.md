<p align="left">
    中文 | <a href="README.md">English</a>
</p>

<p align="center">
    <img src="assets/banner.png" alt="Tiny Qwen" width="90%">
</p>

## ✨ Tiny Qwen

一个用 PyTorch 极简复刻的 Qwen 3.8，自带单文件、无所不能的终端智能体（agentic harness），并支持 int4 量化。

欢迎加入我的 [Discord 频道](https://discord.gg/sBNnqP9gaY)交流！

## 🎇 快速开始

```bash
# 安装依赖
pip install uv
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 用 Qwen 3.8 27B 启动智能体
python run.py
```

也可以通过启动参数选择其他模型或量化：

```bash
python run.py Qwen/Qwen3.5-4B            # 任意 HF repo id，按需下载
python run.py Qwen/Qwen3.5-27B --bits 4  # 下载后量化为 int4 再运行
python run.py weights/Qwen3.5-27B-int4   # 任意本地目录，例如量化后的
```

或完全绕过本地模型，把同一个智能体接到任何 OpenAI 兼容接口上：

```bash
python run.py --url https://api.fireworks.ai/inference/v1/chat/completions \
              --api-key $YOUR_KEY --model accounts/fireworks/models/kimi-k3
```

## 📚 作为库使用

```python
from tiny_qwen import Model, Processor

model = Model.from_pretrained("weights/Qwen3.5-4B")
processor = Processor.from_pretrained("weights/Qwen3.5-4B")

messages = [
    {"role": "user", "content": [
        {"type": "text", "text": "用一句话描述这张图片。"},
        {"type": "image", "image": "photos/cat.jpg"},
    ]},
]
inputs = processor(messages, add_generation_prompt=True, device="mps")

for token_id in model.generate_stream(
    **inputs, max_new_tokens=256, stop_tokens=processor.stop_tokens
):
    print(processor.tokenizer.decode([token_id]), end="", flush=True)
```

## 旧版本

`main` 分支支持整个 Qwen 3.5 / 3.6 / 3.8 架构世代。更早的架构在这些分支：[Qwen 3 VL](https://github.com/Emericen/tiny-qwen/tree/legacy/qwen3_vl)、[Qwen 3 与 Qwen 2.5 VL](https://github.com/Emericen/tiny-qwen/tree/legacy/qwen2_5)。

## 许可证

MIT
