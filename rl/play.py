"""
Baseline: how does a stock model play before any training?

    python -m rl.play --model weights/Qwen3.5-0.8B --hands 20 --vs station --show 1

Prints bb/100 against a scripted fish, the invalid-action rate, and (with
--show N) the full transcript of the first N hands so you can see *how* it
plays, not just the number. The same prompt format is used by rl/train.py.
"""

import argparse
import json
import time
import urllib.request

from rl.game.dealer import BB, FishSeat, play_match
from rl.toolcall import ACT_TOOL, SYSTEM, ToolSeat, configure_tool_parsing, parse_message


TERSE = " Reply with only the tool call — no analysis before it."


def messages_for(observation, system=SYSTEM):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": observation},
    ]


def hf_generator(model_path, device=None, max_new_tokens=128, thinking=False, temperature=1.0):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float16 if device == "mps" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    configure_tool_parsing(tokenizer)
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=dtype)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    model.to(device)
    model.eval()

    def generate(observation):
        text = tokenizer.apply_chat_template(
            messages_for(observation), tools=[ACT_TOOL], tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=1.0,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return parse_message(tokenizer, new_tokens.tolist(), inputs["input_ids"][0].tolist())

    return generate


def vllm_generator(model_path, max_new_tokens=128, thinking=False, temperature=1.0):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, max_model_len=4096, gpu_memory_utilization=0.6)
    tokenizer = llm.get_tokenizer()
    configure_tool_parsing(tokenizer)
    params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=1.0)

    def generate(observation):
        out = llm.chat(
            messages_for(observation), params, tools=[ACT_TOOL], chat_template_kwargs={"enable_thinking": thinking}
        )
        return parse_message(tokenizer, list(out[0].outputs[0].token_ids), list(out[0].prompt_token_ids))

    return generate


def openai_generator(url, model, api_key="", max_new_tokens=128, temperature=1.0, thinking=False, system=SYSTEM):
    """Any OpenAI-compatible chat endpoint: a frontier model, a local server, or OpenMNK's /v1.

    Every response's token usage is accumulated on `generate.usage` — paid APIs
    bill per token, so the caller can show a live meter and enforce a budget."""
    usage = {"in": 0, "out": 0}

    def generate(observation):
        body = {
            "model": model,
            "messages": messages_for(observation, system),
            "tools": [ACT_TOOL],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            # vLLM honors this, keeping server evals on the same thinking setting as training.
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.load(response)
        u = data.get("usage") or {}
        usage["in"] += int(u.get("prompt_tokens") or 0)
        usage["out"] += int(u.get("completion_tokens") or 0)
        return data["choices"][0]["message"]

    generate.usage = usage
    return generate


def showing(generate, hands_to_show):
    """Wrap a generator so the first N hands' observations and replies are printed."""
    state = {"hand": 0}

    def generate_and_print(observation):
        message = generate(observation)
        if observation.startswith("Heads-up") and "blinds posted." in observation:
            state["hand"] += 1
        if state["hand"] <= hands_to_show:
            print("\n--- observation ---\n" + observation)
            shown = {k: message.get(k) for k in ("content", "tool_calls") if message.get(k)}
            print("--- reply ---\n" + json.dumps(shown, indent=2, default=str))
        return message

    return generate_and_print


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="weights/Qwen3.5-0.8B")
    parser.add_argument("--backend", default="hf", choices=["hf", "vllm", "openai"])
    parser.add_argument("--url", default="", help="openai backend: chat completions URL")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--vs", default="station", choices=["station", "nit", "maniac", "random"])
    parser.add_argument("--hands", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--talk", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=768, help="tight caps silently truncate big-pot decisions into forced folds (the 8/24 -470 bb/100 artifact)")
    parser.add_argument("--terse", action="store_true", help="ask for the bare tool call — verbose API models bill every word of analysis")
    parser.add_argument("--token-budget", type=int, default=0, help="stop cleanly (partial stats intact) once total in+out tokens exceed this; 0 = no cap")
    parser.add_argument("--show", type=int, default=1, help="print transcripts of the first N hands")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    system = SYSTEM + TERSE if args.terse else SYSTEM
    if args.backend == "hf":
        generate = hf_generator(args.model, args.device, args.max_new_tokens, args.thinking, args.temperature)
    elif args.backend == "vllm":
        generate = vllm_generator(args.model, args.max_new_tokens, args.thinking, args.temperature)
    else:
        generate = openai_generator(args.url, args.model, args.api_key, args.max_new_tokens, args.temperature, args.thinking, system)
    usage = getattr(generate, "usage", None)
    generate = showing(generate, args.show)

    # Every hand prints a flushed running line: partial results survive any kill,
    # and a paid API's token meter is visible while the money is being spent.
    def progress(done, total_hands, chips):
        meter = f"  tokens {usage['in']}+{usage['out']}" if usage else ""
        print(f"hand {done}/{total_hands}  running {chips / BB:+.1f} bb{meter}", flush=True)
        if args.token_budget and usage and usage["in"] + usage["out"] > args.token_budget:
            print(f"token budget {args.token_budget} exhausted — stopping with partial results", flush=True)
            return False

    policy = ToolSeat(generate)
    start = time.time()
    stats = play_match(
        lambda: policy, lambda: FishSeat(args.vs, seed=args.seed), hands=args.hands, seed=args.seed, talk=args.talk, progress=progress
    )
    elapsed = time.time() - start
    print(flush=True)
    print(f"model      {args.model}")
    print(f"opponent   {args.vs}")
    print(f"hands      {stats['hands']} (mirrored)")
    print(f"bb/100     {stats['bb_per_100']:+.1f}")
    print(f"invalid    {100 * stats['invalid_rate_a']:.1f}% of {policy.decisions} decisions")
    if usage:
        print(f"tokens     {usage['in']} in, {usage['out']} out")
    print(f"time       {elapsed:.0f}s ({elapsed / max(1, policy.decisions):.2f}s per decision)", flush=True)


if __name__ == "__main__":
    main()
