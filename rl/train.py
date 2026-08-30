"""
GRPO on poker: the dataset is a list of deals.

    python -m rl.train --model Qwen/Qwen3.8-27B --steps 300

The three pieces stay separate and this file only wires them:

    the game     rl/poker.py  — rules, unaware it is being trained on
    the env      PokerEnv     — one hand as a tool-calling episode
    the trainer  TRL          — rollouts, loss, gradients, vLLM

TRL calls PokerEnv.reset(**row) once per rollout, exposes act() to the model
as a tool (the schema is derived from its signature and docstring), and reads
get_reward() when the episode ends. Because every row is a SEED and GRPO draws
`num_generations` rollouts per row, all G rollouts of a group replay the SAME
deal — so the group-mean baseline subtracts card luck and the advantage
measures only how the hand was played. That is the whole trick; the rest is
configuration.
"""

import argparse
import random

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from rl.poker import BIG_BLIND, Game, render

OPENERS = "You are playing no-limit Texas hold'em for chips. Play the hand."

FISH = {  # scripted opponents; a policy is a function of the hand, nothing more
    "station": lambda hand, rng: hand.legal_totals()[0],
    "nit": lambda hand, rng: (None if hand.legal_totals()[0] > hand.bets[hand.acting_seat]
                              else hand.legal_totals()[0]),
    "maniac": lambda hand, rng: (hand.legal_totals()[1].start if hand.legal_totals()[1]
                                 else hand.legal_totals()[0]),
}


class PokerEnv:
    """One hand, as an episode. reset() deals and returns the first
    observation; act() is the model's only tool; get_reward() pays out in bb.

    TRL pools and REUSES instances across batches, so reset() must rebuild
    every piece of state rather than assume a fresh object."""

    def __init__(self, opponents=("station", "nit", "maniac")):
        self.opponents = opponents

    def reset(self, seed: int, **_) -> str:
        """Deal the hand named by `seed`. Same seed = same cards, so the G
        rollouts of one group differ only in how the model plays."""
        self.rng = random.Random(seed)
        style = self.opponents[seed % len(self.opponents)]
        self.opponent = FISH[style]
        # The button is always seat 0; alternating WHICH seat we occupy is what
        # gives both positions coverage. Both are functions of the seed, so they
        # are constant across a group — only our own decisions vary inside it.
        self.seat = seed % 2
        names = ["you", f"fish ({style})"] if self.seat == 0 else [f"fish ({style})", "you"]
        self.game = Game(names, chips=200, seed=seed, talk=False)
        self.done = False
        self._let_opponent_play()
        # A nit in the small blind folds before we ever act: that hand carries a
        # reward with no decision to attribute it to. Deal on until we face one.
        # Deterministic in the seed, so a group still shares its starting point.
        for _ in range(20):
            if not self.done:
                break
            self.game.new_hand()
            self.done = False
            self._let_opponent_play()
        else:
            return None  # pathological seed; TRL leaves the prompt unchanged
        self.start = self.game.stacks[self.seat]
        return render(self.game.observe(self.seat))

    def act(self, total: int | None) -> str:
        """Commit to a cumulative chip total for this hand, or null to fold.
        Pick one of the legal totals shown in the table state."""
        if self.done:
            return "The hand is over."
        try:
            self.game.act(total)
        except ValueError:
            legal = self.game.observe(self.seat)["legal"]
            allowed = ", ".join("null" if e["total"] is None else str(e["total"]) for e in legal)
            return f"{total} is not a legal total here. Legal totals: {allowed}. Try again."
        self._let_opponent_play()
        if self.done:
            delta = self.game.stacks[self.seat] - self.start
            return f"Hand over. You {'won' if delta > 0 else 'lost'} {abs(delta)} chips."
        return render(self.game.observe(self.seat))

    def get_reward(self) -> float:
        """The verifiable signal: chips won or lost this hand, in big blinds."""
        return (self.game.stacks[self.seat] - self.start) / BIG_BLIND

    def _let_opponent_play(self):
        """Run the table until it is our turn again, or the hand ends."""
        while True:
            hand = self.game.hand
            if hand.acting_seat is None:
                self.done = True
                return
            if hand.acting_seat == self.seat:
                return
            self.game.act(self.opponent(hand, self.rng))


def deals(n: int, start: int = 0) -> Dataset:
    """The dataset is seeds. One row is one deal; GRPO replays each G times."""
    return Dataset.from_list(
        [{"prompt": [{"role": "user", "content": OPENERS}], "seed": s}
         for s in range(start, start + n)]
    )


def main():
    parser = argparse.ArgumentParser(description="GRPO on poker")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--deals", type=int, default=20000)
    parser.add_argument("--gens", type=int, default=8, help="G: rollouts per deal")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--loss", default="dapo", help="grpo | dr_grpo | dapo | gspo | cispo | ...")
    parser.add_argument("--vllm-mem", type=float, default=0.45)
    parser.add_argument("--out", default="runs/poker")
    args = parser.parse_args()

    config = GRPOConfig(
        output_dir=args.out,
        num_generations=args.gens,          # G rollouts of the SAME deal
        scale_rewards=False,                # Dr. GRPO: chip std tracks pot size, not skill
        loss_type=args.loss,
        beta=0.0,                           # no KL leash: the reward is verifiable
        mask_truncated_completions=True,    # DAPO
        learning_rate=args.lr,
        max_steps=args.steps,
        use_vllm=True,
        vllm_gpu_memory_utilization=args.vllm_mem,
        logging_steps=1,
        save_steps=25,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=deals(args.deals),
        environment_factory=PokerEnv,       # reset / act (a tool) / get_reward
        peft_config=LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank,
                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
    trainer.train()


if __name__ == "__main__":
    main()
