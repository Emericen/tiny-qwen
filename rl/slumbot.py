"""
Play against Slumbot (slumbot.com), the free public heads-up NLHE bot.

    python -m rl.slumbot --hands 20 --policy fish            # sanity: a calling station vs Slumbot
    python -m rl.slumbot --hands 20 --model weights/Qwen3.5-0.8B

Slumbot is the fixed external anchor: it never changes, so a bb/100 number
against it from today can be compared with one from next month. It plays
200 bb deep with blinds 50/100; we divide every amount by 50 so the model
sees the same 1/2-chip units it was trained on (stacks show as 400).

Protocol (from slumbot.com's sample client): POST /api/new_hand, then POST
/api/act with {"token", "incr"}. Action strings: k check, c call, f fold,
b<N> bet-to N on this street, '/' between streets. client_pos 1 = button.
"""

import argparse
import json
import urllib.request

from rl.game.dealer import FishSeat, STREETS, cards_str, parse_card
from rl.toolcall import ToolSeat

HOST = "https://slumbot.com"
SB = 50
BB = 100
STACK = 20000
SCALE = 50  # Slumbot chips per one of our chips


def post(path, body):
    request = urllib.request.Request(
        HOST + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


class SlumbotState:
    """Replays Slumbot's action string into pot, stacks, history and legal actions."""

    def __init__(self, action, client_pos):
        self.client = client_pos  # 1 = button / small blind
        self.street = 0
        self.bet = [BB, SB]  # per position this street: pos 0 = big blind, pos 1 = button
        self.invested = [BB, SB]
        self.last_bet_size = BB - SB
        self.pos = 1  # button acts first preflop
        self.ends_street = False  # a check/call now would close the street
        self.over = False
        self.history = []
        self._replay(action)

    def _next_street(self):
        self.street += 1
        self.bet = [0, 0]
        self.last_bet_size = 0
        self.pos = 0  # big blind acts first after the flop
        self.ends_street = False

    def _replay(self, action):
        i = 0
        while i < len(action):
            c = action[i]
            i += 1
            if c == "/":
                continue
            if c == "f":
                self.history.append((self.street, self.pos, "fold"))
                self.over = True
                return
            if c == "k":
                self.history.append((self.street, self.pos, "check"))
                if self.ends_street:
                    if self.street == 3:
                        self.over = True
                        return
                    self._next_street()
                else:
                    self.ends_street = True
                    self.pos = 1 - self.pos
                continue
            if c == "c":
                amount = max(self.bet) - self.bet[self.pos]
                self.bet[self.pos] += amount
                self.invested[self.pos] += amount
                self.history.append((self.street, self.pos, "call"))
                if max(self.invested) == STACK or self.street == 3:
                    self.over = self.ends_street or max(self.invested) == STACK
                    if self.over:
                        return
                if self.ends_street:
                    self._next_street()
                else:
                    self.ends_street = True
                    self.pos = 1 - self.pos
                continue
            if c == "b":
                j = i
                while i < len(action) and action[i].isdigit():
                    i += 1
                to = int(action[j:i])
                self.last_bet_size = to - max(self.bet)
                self.invested[self.pos] += to - self.bet[self.pos]
                self.bet[self.pos] = to
                verb = "bet" if max(self.bet) == to and self.bet[1 - self.pos] == 0 else "raise"
                self.history.append((self.street, self.pos, f"{verb} {to / SCALE:g}"))
                self.ends_street = True
                self.pos = 1 - self.pos
                continue
            raise ValueError(f"unexpected character {c!r} in action {action!r}")

    # --- what the model sees ---

    def to_call(self):
        return max(self.bet) - self.bet[self.client]

    def legal(self):
        """Our-unit action strings plus the Slumbot 'incr' each maps to."""
        call = self.to_call()
        options = []
        if call > 0:
            options.append(("fold", "f"))
            options.append(("call", "c"))
        else:
            options.append(("check", "k"))
        street_start = self.invested[self.client] - self.bet[self.client]
        max_to = STACK - street_start
        min_to = max(self.bet) + max(self.last_bet_size, BB)
        if max(self.bet) == max_to or min_to >= max_to:
            if max(self.bet) < max_to:
                options.append(("allin", f"b{max_to}"))
            return options
        pot_now = sum(self.invested) + call
        verb = "bet" if max(self.bet) == 0 else "raise"
        sizes = [min_to]
        for frac in (0.5, 1.0, 2.0):
            sizes.append(max(self.bet) + int(round(frac * pot_now)))
        for to in sorted(set(sizes)):
            if min_to <= to < max_to:
                options.append((f"{verb} {to / SCALE:g}", f"b{to}"))
        options.append(("allin", f"b{max_to}"))
        return options

    def observation(self, hole_cards, board):
        me = self.client
        opp = 1 - me
        role = "button (small blind)" if me == 1 else "big blind"
        lines = []
        lines.append(f"Heads-up no-limit hold'em. Blinds 1/2. 1 bb = 2 chips. Starting stacks {STACK // SCALE}.")
        lines.append(f"You are seat {me}, the {role}.")
        lines.append(f"Your hole cards: {' '.join(hole_cards)}")
        board_text = " ".join(board) if board else "(none yet)"
        lines.append(f"Street: {STREETS[self.street]}. Board: {board_text}")
        pot = sum(self.invested) / SCALE
        lines.append(
            f"Pot: {pot:g}. Your stack: {(STACK - self.invested[me]) / SCALE:g}. "
            f"Opponent stack: {(STACK - self.invested[opp]) / SCALE:g}."
        )
        parts = []
        last_street = None
        for street, who, text in self.history:
            if street != last_street:
                parts.append(f"[{STREETS[street]}]")
                last_street = street
            parts.append(("you " if who == me else "opp ") + text)
        lines.append("Action so far: " + (" ".join(parts) if parts else "blinds posted."))
        call = self.to_call()
        if call > 0:
            lines.append(f"To call: {call / SCALE:g}.")
        lines.append("Legal actions: " + ", ".join(a for a, _ in self.legal()))
        lines.append("Reply with exactly one legal action from the list.")
        return "\n".join(lines)


def play_hand(seat, token, show=False):
    r = post("/api/new_hand", {"token": token} if token else {})
    token = r.get("token", token)
    while True:
        if "winnings" in r and r["winnings"] is not None:
            return token, r["winnings"], r.get("baseline_winnings")
        if "error_msg" in r:
            raise RuntimeError(r["error_msg"])
        state = SlumbotState(r["action"], r["client_pos"])
        obs = state.observation(r["hole_cards"], r["board"])
        options = state.legal()
        action, _ = seat.act(obs, [a for a, _ in options])
        incr = dict(options)[action]
        if show:
            print("\n--- observation ---\n" + obs)
            print(f"--- action --- {action}  (incr {incr})")
        r = post("/api/act", {"token": token, "incr": incr})
        token = r.get("token", token)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=10)
    parser.add_argument("--policy", default="model", choices=["model", "fish"])
    parser.add_argument("--vs-style", default="station", help="fish style when --policy fish")
    parser.add_argument("--model", default="weights/Qwen3.5-0.8B")
    parser.add_argument("--backend", default="hf", choices=["hf", "vllm", "openai"])
    parser.add_argument("--url", default="", help="openai backend: chat completions URL")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=768, help="tight caps silently truncate big-pot decisions into forced folds")
    parser.add_argument("--terse", action="store_true", help="ask for the bare tool call — verbose API models bill every word of analysis")
    parser.add_argument("--token-budget", type=int, default=0, help="stop cleanly (partial stats intact) once total in+out tokens exceed this; 0 = no cap")
    parser.add_argument("--show", type=int, default=1)
    args = parser.parse_args()

    if args.policy == "fish":
        seat = FishSeat(args.vs_style, seed=0)
    else:
        from rl.play import SYSTEM, TERSE, hf_generator, openai_generator, vllm_generator

        system = SYSTEM + TERSE if args.terse else SYSTEM
        if args.backend == "hf":
            generate = hf_generator(args.model, max_new_tokens=args.max_new_tokens, thinking=args.thinking)
        elif args.backend == "openai":
            generate = openai_generator(args.url, args.model, args.api_key, args.max_new_tokens, thinking=args.thinking, system=system)
        else:
            generate = vllm_generator(args.model, max_new_tokens=args.max_new_tokens, thinking=args.thinking)
        seat = ToolSeat(generate)

    usage = getattr(getattr(seat, "generate", None), "usage", None)
    token = None
    total = 0
    baseline_total = 0
    played = 0
    for i in range(args.hands):
        token, winnings, baseline = play_hand(seat, token, show=i < args.show)
        total += winnings
        baseline_total += baseline or 0
        played += 1
        meter = f"  tokens {usage['in']}+{usage['out']}" if usage else ""
        print(f"hand {played}: {winnings / BB:+.1f} bb  running {total / BB:+.1f} bb{meter}", flush=True)
        if args.token_budget and usage and usage["in"] + usage["out"] > args.token_budget:
            print(f"token budget {args.token_budget} exhausted — stopping with partial results", flush=True)
            break
    print(flush=True)
    print(f"hands     {played}")
    print(f"bb/100    {100 * total / max(1, played) / BB:+.1f}")
    print(f"baseline  {100 * baseline_total / max(1, played) / BB:+.1f}  (Slumbot's own variance-reduced number)")
    if hasattr(seat, "decisions"):
        print(f"invalid   {100 * seat.invalid / max(1, seat.decisions):.1f}% of {seat.decisions} decisions", flush=True)


if __name__ == "__main__":
    main()
