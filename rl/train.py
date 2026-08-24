"""
GRPO on heads-up hold'em, using TRL's environment loop.

    python -m rl.train --model Qwen/Qwen3.5-0.8B --vs station --steps 200 --gens 8

The dealer is the environment. The model plays a hand by calling the `act`
tool until the hand ends; the reward is its chip result in big blinds. The
`gens` rollouts of one group all start from the same deal (same seed, same
button), so the group-relative advantage subtracts card luck exactly — the
GRPO analogue of a mirrored deck.

What is NOT here yet, on purpose (curve first): self-play opponents, the
per-role EMA baseline from SPIRAL, table talk in training, thinking mode.
Each is a flag away once this loop moves the invalid rate and bb/100 vs fish.
"""

import argparse

import torch

from rl.game.dealer import BB, Dealer, FishSeat, default_action, parse_reply

SYSTEM = (
    "You are playing heads-up no-limit Texas hold'em for chips. "
    "Play the hand by calling the `act` tool with one legal action copied exactly from the list. "
    "Keep calling `act` after each new game state until the hand is over. "
    "Never answer in plain text; every decision must be an `act` call."
)


class PokerEnv:
    """One hand per episode. Public methods become tools the model can call."""

    opponent = "station"
    talk = False
    equity_samples = 100
    stats = {"episodes": 0, "decisions": 0, "invalid": 0, "unfinished": 0, "chips": 0.0}

    # -- reserved by TRL --

    def reset(self, seed: int = 0, button: int = 0, **kwargs) -> str:
        self.dealer = Dealer(seed, button=button, talk=self.talk, equity_samples=self.equity_samples)
        self.fish = FishSeat(self.opponent, seed=seed)
        self.invalid = 0
        self.decisions = 0
        self._opponent_moves()
        return SYSTEM + "\n\n" + self.dealer.observation(0)

    def get_reward(self) -> float:
        if not self.dealer.done:  # the model stopped calling tools mid-hand: play it out passively
            PokerEnv.stats["unfinished"] += 1
            while not self.dealer.done:
                self.dealer.step(default_action(self.dealer.legal_actions(0)))
                self._opponent_moves()
        chips = self.dealer.result()[0]
        s = PokerEnv.stats
        s["episodes"] += 1
        s["decisions"] += self.decisions
        s["invalid"] += self.invalid
        s["chips"] += chips
        if s["episodes"] % 64 == 0:
            print(
                f"[poker] episodes {s['episodes']}  bb/100 {100 * s['chips'] / s['episodes'] / BB:+.1f}  "
                f"invalid {100 * s['invalid'] / max(1, s['decisions']):.1f}%  "
                f"unfinished {100 * s['unfinished'] / s['episodes']:.1f}%",
                flush=True,
            )
        return chips / BB

    # -- the one tool --

    def act(self, action: str, say: str = "") -> str:
        """
        Take your poker action.

        Args:
            action: One legal action copied exactly from the list, for example "call", "raise 6", "bet 4" or "allin".
            say: Optional table talk your opponent will see.

        Returns:
            The new game state, or the result if the hand is over.
        """
        if self.dealer.done:
            return "The hand is over."
        legal = self.dealer.legal_actions(0)
        self.decisions += 1
        chosen, _ = parse_reply(action, legal)
        if chosen is None:
            self.invalid += 1
            chosen = default_action(legal)
        self.dealer.step(chosen, say)
        self._opponent_moves()
        return self.dealer.observation(0)

    # -- private (underscore = not exposed as a tool) --

    def _opponent_moves(self):
        while not self.dealer.done and self.dealer.to_act == 1:
            obs = self.dealer.observation(1)
            legal = self.dealer.legal_actions(1)
            action, say = self.fish.act(obs, legal)
            self.dealer.step(action, say)


def processing_class_that_parses_tool_calls(model_path):
    """Qwen3.5 emits XML tool calls (<function=act><parameter=action>...). TRL parses them with
    transformers' parse_response, driven by a response template (new style) or a response
    schema (legacy). Some transformers builds accept the template yet return no tool_calls,
    silently — every hand would then resolve as a passive fold and the run would train on
    nothing. So we exercise the exact path the trainer uses and fall back to the legacy schema.

    Qwen3.5 checkpoints are vision-language models; we hand TRL the processor (not just the
    tokenizer) so it reads the text config correctly, even though every prompt here is text."""
    import os

    from transformers import AutoProcessor, AutoTokenizer
    from trl.chat_template_utils import add_response_schema, parse_response, qwen3_5_schema

    if os.path.exists(os.path.join(model_path, "preprocessor_config.json")) or not os.path.isdir(model_path):
        processing_class = AutoProcessor.from_pretrained(model_path)
    else:
        processing_class = AutoTokenizer.from_pretrained(model_path)
    tokenizer = getattr(processing_class, "tokenizer", processing_class)
    add_response_schema(tokenizer)
    tools = [{"type": "function", "function": {"name": "act", "parameters": {"type": "object", "properties": {"action": {"type": "string"}}}}}]
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "call the act tool"}], tools=tools, add_generation_prompt=True, tokenize=True
    )
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids["input_ids"]
    sample = "<think>\n\n</think>\n\n<tool_call>\n<function=act>\n<parameter=action>\ncall\n</parameter>\n</function>\n</tool_call>"
    sample_ids = tokenizer(sample + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

    def parses():
        try:
            return bool(parse_response(tokenizer, sample_ids, prefix=prompt_ids).get("tool_calls"))
        except Exception:
            return False

    if not parses():
        tokenizer.response_template = None
        tokenizer.response_schema = qwen3_5_schema
        print("[poker] using legacy response_schema for tool-call parsing", flush=True)
    if not parses():
        raise RuntimeError("tokenizer cannot parse Qwen3.5 tool calls; training would never execute an action")
    return processing_class


def build_dataset(n_deals):
    """One row per deal. Every rollout in a group shares a row, hence a deck."""
    from datasets import Dataset

    rows = []
    env = PokerEnv()
    for i in range(n_deals):
        prompt = env.reset(seed=i, button=i % 2)
        rows.append({"prompt": [{"role": "user", "content": prompt}], "seed": i, "button": i % 2})
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--vs", default="station", choices=["station", "nit", "maniac", "random"])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--gens", type=int, default=8, help="rollouts per deal (GRPO group size)")
    parser.add_argument("--deals", type=int, default=4096, help="distinct deals in the dataset")
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max-tokens", type=int, default=1024, help="completion budget for the whole hand")
    parser.add_argument("--lora-r", type=int, default=16, help="0 = full-parameter")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--talk", action="store_true")
    parser.add_argument("--vllm", default="colocate", choices=["colocate", "server", "none"])
    parser.add_argument("--vllm-host", default="localhost")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-mem", type=float, default=0.3, help="colocate: vLLM's share of GPU memory; 27B bf16 weights need ~0.45 of an H200")
    parser.add_argument("--output", default="runs/poker")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--report-to", default="none", help="none | wandb | tensorboard")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from trl import GRPOConfig, GRPOTrainer

    PokerEnv.opponent = args.vs
    PokerEnv.talk = args.talk

    config = GRPOConfig(
        output_dir=args.output,
        max_steps=args.steps,
        num_generations=args.gens,
        per_device_train_batch_size=args.gens,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        temperature=1.0,
        max_completion_length=args.max_tokens,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=args.save_every,
        report_to=args.report_to,
        seed=args.seed,
        use_vllm=args.vllm != "none",
        vllm_mode=args.vllm if args.vllm != "none" else "colocate",
        vllm_server_host=args.vllm_host,
        vllm_server_port=args.vllm_port,
        vllm_gpu_memory_utilization=args.vllm_mem,
        log_completions=True,
        num_completions_to_print=1,
        chat_template_kwargs={"enable_thinking": args.thinking},
    )

    peft_config = None
    if args.lora_r > 0:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"
        )

    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        processing_class=processing_class_that_parses_tool_calls(args.model),
        environment_factory=PokerEnv,
        train_dataset=build_dataset(args.deals),
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output + "/final")
    s = PokerEnv.stats
    print(f"done. episodes {s['episodes']}  bb/100 {100 * s['chips'] / max(1, s['episodes']) / BB:+.1f}")


if __name__ == "__main__":
    main()
