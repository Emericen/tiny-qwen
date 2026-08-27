"""
Poker in two files: this module (the whole game) and table.html (the glass).

    python -m rl.game                                    # you vs a calling station
    python -m rl.game --p0 maniac --p1 nit               # spectate two fish
    python -m rl.game --p1 api:grok-4.5@https://api.x.ai/v1 --key1 $XAI_API_KEY

Layers, top to bottom, with a hard rule — nothing below imports anything above:

    table.html   presentation: reads JSON, posts clicks; knows no Python
    bridge       (inside main) stdlib http.server: objects -> JSON, clicks -> step()
    Table        one browser session: seats, pacing, narration
    Dealer       the game across hands: chips, button, score, the chat thread
    Hand         one episode: cards, betting, settlement — the rules
    scoring      pure math: 5 and 7 card evaluation

The engine speaks ONE action: your cumulative chip total for the hand
(None = fold). check / call / bet / raise / all-in are presentation-layer
words for particular numbers. Cards are emoji strings ("♠A") for humans and
converted to ASCII ("As") at the model/browser boundary.

DEVIATION (documented): every all-in re-opens the action, even below a full
raise — real poker's incomplete-raise rule is dropped for legibility.
"""

import itertools
import json
import random
import re
import threading
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# the deck
# ---------------------------------------------------------------------------

SMALL_BLIND, BIG_BLIND = 1, 2
CARDS_REVEALED = (0, 3, 4, 5)  # per street: preflop, flop, turn, river
RANKS = "23456789TJQKA"
SUITS = "♣♦♥♠"

INT_TO_CARD = {c: SUITS[c % 4] + ("10" if c // 4 == 8 else RANKS[c // 4]) for c in range(52)}
CARD_TO_INT = {v: k for k, v in INT_TO_CARD.items()}
SUIT_ASCII = {"♣": "c", "♦": "d", "♥": "h", "♠": "s"}


def card_ascii(card: str) -> str:
    """'♠A' -> 'As', '♥10' -> 'Th' — the format models and table.html speak."""
    rank = card[1:]
    return ("T" if rank == "10" else rank) + SUIT_ASCII[card[0]]


# ---------------------------------------------------------------------------
# scoring: higher number = better hand, equal = split
# ---------------------------------------------------------------------------


def get_5_score(cards) -> int:
    ranks = [CARD_TO_INT[card] // 4 for card in cards]
    suits = [CARD_TO_INT[card] % 4 for card in cards]
    cnt = Counter(ranks)
    tiebreaker = sorted(ranks, key=lambda r: (cnt[r], r), reverse=True)
    counts = sorted(cnt.values(), reverse=True)
    unique = sorted(set(ranks))
    is_flush = len(set(suits)) == 1
    is_straight = len(unique) == 5 and unique[4] - unique[0] == 4
    if set(ranks) == {0, 1, 2, 3, 12}:  # wheel: A-2-3-4-5
        is_straight, tiebreaker = True, [3, 2, 1, 0, -1]  # ace plays low
    # fmt: off
    if is_straight and is_flush: score = 8
    elif counts == [4, 1]:       score = 7
    elif counts == [3, 2]:       score = 6
    elif is_flush:               score = 5
    elif is_straight:            score = 4
    elif counts == [3, 1, 1]:    score = 3
    elif counts == [2, 2, 1]:    score = 2
    elif counts == [2, 1, 1, 1]: score = 1
    else:                        score = 0
    # fmt: on
    for t in tiebreaker:
        score = score * 14 + (t + 1)  # shift -1..12 into base-14 digits
    return score


def get_7_score(cards) -> int:
    """Best 5-card score out of 7 cards."""
    return max(get_5_score(combo) for combo in itertools.combinations(cards, 5))


# ---------------------------------------------------------------------------
# Hand: one episode — the rules
# ---------------------------------------------------------------------------


class Hand:
    """one episode, ANY number of players. An action is ONE number: your
    cumulative chip total for the hand. None = fold. bets[] is each player's
    total investment; the pot is its sum; side pots are layers of it. A street
    ends when everyone still able to act has answered the current price."""

    def __init__(self, stacks: list[int], seed: int | None = None):
        self.player_count = len(stacks)
        self.board, self.holes = self._deal(self.player_count, seed)
        self.stacks = stacks.copy()
        self.bets = [0] * self.player_count
        self.folded = [False] * self.player_count
        self.min_raise = BIG_BLIND
        self.street = 0
        small_blind_seat, big_blind_seat = (0, 1) if self.player_count == 2 else (1, 2)
        self._commit(small_blind_seat, SMALL_BLIND)
        self._commit(big_blind_seat, BIG_BLIND)
        self.answers_owed = sum(not self.is_all_in(seat) for seat in range(self.player_count))
        self.acting_seat = self._next_able_seat(big_blind_seat)  # None = hand over
        if self.acting_seat is None:  # blinds put everyone all-in
            self._settle()

    def is_all_in(self, seat: int) -> bool:
        return self.stacks[seat] == 0

    def revealed(self) -> list[str]:
        return self.board[: CARDS_REVEALED[self.street]]

    def _commit(self, seat: int, amount: int):
        amount = min(amount, self.stacks[seat])
        self.stacks[seat] -= amount
        self.bets[seat] += amount

    def _next_able_seat(self, after_seat: int) -> int | None:
        for steps in range(1, self.player_count + 1):
            seat = (after_seat + steps) % self.player_count
            if not self.folded[seat] and not self.is_all_in(seat):
                return seat
        return None

    def legal_totals(self) -> tuple[int, range]:
        """The two legal kinds of commit_to for the acting seat:
        (match, escalation) — the total that keeps you in, and the range of
        legal escalation totals (empty when escalation is impossible)."""
        seat = self.acting_seat
        my_all_in = self.bets[seat] + self.stacks[seat]
        biggest_opponent_all_in = max(
            self.bets[other] + self.stacks[other]
            for other in range(self.player_count)
            if other != seat and not self.folded[other]
        )
        price = max(self.bets)
        match = min(price, my_all_in)
        escalation_cap = min(my_all_in, biggest_opponent_all_in)
        if escalation_cap > price:
            escalation = range(min(price + self.min_raise, escalation_cap), escalation_cap + 1)
        else:
            escalation = range(0)
        return match, escalation

    def step(self, commit_to: int | None):
        seat = self.acting_seat
        if seat is None:
            raise ValueError("hand is over")
        if commit_to is None:  # fold
            self.folded[seat] = True
            self.answers_owed -= 1
            if self.folded.count(False) == 1:  # everyone else gave up
                return self._settle()
        else:
            match, escalation = self.legal_totals()
            if commit_to != match and commit_to not in escalation:
                raise ValueError(f"{commit_to}: legal totals are {match} or {escalation}")
            price = max(self.bets)
            if commit_to > price:  # escalation re-asks everyone
                if commit_to - price >= self.min_raise:
                    self.min_raise = commit_to - price
                self._commit(seat, commit_to - self.bets[seat])
                self.answers_owed = sum(
                    not self.folded[other] and not self.is_all_in(other)
                    for other in range(self.player_count)
                    if other != seat
                )
            else:  # match answers the price
                self._commit(seat, commit_to - self.bets[seat])
                self.answers_owed -= 1
        if self.answers_owed <= 0:
            self._next_street()
        else:
            self.acting_seat = self._next_able_seat(seat)

    def _next_street(self):
        able_to_act = sum(
            not self.folded[seat] and not self.is_all_in(seat)
            for seat in range(self.player_count)
        )
        if self.street == 3 or able_to_act < 2:
            return self._settle()  # showdown / run it out
        self.street += 1
        self.min_raise = BIG_BLIND
        self.answers_owed = able_to_act
        self.acting_seat = self._next_able_seat(0)  # first able seat after the button

    def _settle(self):
        """award the pot in layers — each layer capped by its shortest investor"""
        self.street = 3
        live = [seat for seat in range(self.player_count) if not self.folded[seat]]
        if len(live) > 1:
            scores = {seat: get_7_score(self.holes[seat] + self.board) for seat in live}
        else:
            scores = {live[0]: 1}
        remaining = live[:]
        while remaining and max(self.bets) > 0:
            cap = min(self.bets[seat] for seat in remaining)
            layer = sum(min(bet, cap) for bet in self.bets)
            self.bets = [bet - min(bet, cap) for bet in self.bets]
            best_score = max(scores[seat] for seat in remaining)
            winners = [seat for seat in remaining if scores[seat] == best_score]
            share = layer // len(winners)
            for winner in winners:
                self.stacks[winner] += share
            self.stacks[winners[0]] += layer - share * len(winners)  # odd chip
            remaining = [seat for seat in remaining if self.bets[seat] > 0]
        for seat, bet in enumerate(self.bets):  # unmatched residue refunds itself
            self.stacks[seat] += bet
        self.bets = [0] * self.player_count
        self.acting_seat = None

    @staticmethod
    def _deal(player_count: int, seed: int | None):
        rng = random.Random(seed)
        deck = list(INT_TO_CARD.values())
        rng.shuffle(deck)
        return deck[:5], [deck[5 + 2 * i : 7 + 2 * i] for i in range(player_count)]


# ---------------------------------------------------------------------------
# presentation vocabulary: numbers -> poker words (and back)
# ---------------------------------------------------------------------------


def labeled_legal(hand: Hand) -> list[dict]:
    """The acting seat's menu as poker words. Used by the browser buttons AND
    by model observations, so both speak the same vocabulary."""
    seat = hand.acting_seat
    match, escalation = hand.legal_totals()
    cost = match - hand.bets[seat]
    menu = []
    if cost > 0:
        menu.append({"action": "fold", "label": "Fold", "n": None})
        menu.append({"action": "call", "label": "Call", "n": cost})
    else:
        menu.append({"action": "check", "label": "Check", "n": None})
    if escalation:
        price = max(hand.bets)
        pot_now = sum(hand.bets) + cost
        all_in = escalation[-1]
        sizes = {escalation.start: "Min"}
        for fraction, label in ((0.5, "½ pot"), (1.0, "Pot"), (2.0, "2× pot")):
            n = price + round(fraction * pot_now)
            if n in escalation:
                sizes.setdefault(n, label)
        for n, label in sorted(sizes.items()):
            if n != all_in:
                menu.append({"action": f"raise {n}", "label": label, "n": n})
        menu.append({"action": "allin", "label": "All-in", "n": all_in})
    return menu


def action_to_total(hand: Hand, action: str) -> int | None:
    """A menu action string -> the commit_to number the engine speaks."""
    match, escalation = hand.legal_totals()
    action = action.strip().lower()
    if action == "fold":
        return None
    if action in ("check", "call"):
        return match
    if action == "allin":
        return escalation[-1] if escalation else match
    return int(action.split()[1])  # "raise N" — engine validates N


ACTION_RE = re.compile(r"\b(fold|check|call|all-?in|raise\s+\d+|bet\s+\d+)\b", re.IGNORECASE)
SAY_RE = re.compile(r"say:\s*(.+)", re.IGNORECASE)


def parse_reply(text: str, hand: Hand) -> tuple[int | None, str, bool]:
    """Free model text -> (commit_to, say, valid). The last action mentioned
    wins; no valid action -> the cheapest legal thing (check if free, else
    fold), reported as valid=False so harnesses can count it."""
    seat = hand.acting_seat
    match, escalation = hand.legal_totals()
    say = ""
    said = SAY_RE.search(text)
    if said:
        say = said.group(1).strip().strip('"')
        text = text[: said.start()]
    chosen, found = None, False
    for hit in ACTION_RE.finditer(text):
        word = re.sub(r"\s+", " ", hit.group(1).lower())
        if word in ("check", "call"):
            chosen, found = match, True
        elif word in ("allin", "all-in"):
            chosen, found = (escalation[-1] if escalation else match), True
        elif word == "fold":
            chosen, found = None, True
        else:
            n = int(word.split()[1])
            if n in escalation:
                chosen, found = n, True
    if not found:
        cost = match - hand.bets[seat]
        chosen = None if cost > 0 else match
    return chosen, say, found


def observation(hand: Hand, seat: int, name: str, chat: list, talk: bool) -> str:
    """What THIS player may know, as text. ASCII cards — the model's format."""
    lines = []
    kind = "Heads-up" if hand.player_count == 2 else f"{hand.player_count}-player"
    lines.append(f"{kind} no-limit hold'em. Blinds {SMALL_BLIND}/{BIG_BLIND}. 1 bb = {BIG_BLIND} chips.")
    lines.append(f"You are {name} (seat {seat}).")
    lines.append(f"Your hole cards: {' '.join(card_ascii(c) for c in hand.holes[seat])}")
    board = " ".join(card_ascii(c) for c in hand.revealed()) or "(none yet)"
    lines.append(f"Street: {['preflop', 'flop', 'turn', 'river'][hand.street]}. Board: {board}")
    lines.append(f"Pot: {sum(hand.bets)}. Your stack: {hand.stacks[seat]}.")
    menu = labeled_legal(hand)
    for entry in menu:
        if entry["action"] == "call":
            lines.append(f"To call: {entry['n']}.")
    lines.append("Legal actions: " + ", ".join(entry["action"] for entry in menu))
    lines.append("Reply with exactly one legal action from the list.")
    if talk:
        recent = [entry for entry in chat if entry["kind"] == "talk"][-8:]
        if recent:
            lines.append("Table talk so far:")
            lines.extend(f'{entry["who"]}: {entry["text"]}' for entry in recent)
        lines.append("You may add table talk on a second line: say: <message>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# seats: things that turn an observation into an action
# ---------------------------------------------------------------------------

FISH_STYLES = ("station", "nit", "maniac", "random")

SYSTEM = (
    "You are playing no-limit Texas hold'em for chips. "
    "Read the state and reply with exactly one legal action copied from the list. "
    "You may add one more line starting with 'say:' to talk to the table."
)


class FishSeat:
    """Scripted opponents. Exploitable on purpose; they are the beta meter."""

    def __init__(self, style: str = "station", seed: int = 0):
        self.style = style
        self.rng = random.Random(seed)

    def act(self, observation_text: str, hand: Hand) -> tuple[int | None, str]:
        match, escalation = hand.legal_totals()
        cost = match - hand.bets[hand.acting_seat]
        if self.style == "station":  # calls everything, never raises
            return match, ""
        if self.style == "nit":  # folds to any bet, checks otherwise
            return (None if cost > 0 else match), ""
        if self.style == "maniac":  # escalates when it can, else calls
            if escalation:
                return self.rng.choice([escalation.start, escalation[len(escalation) // 2], escalation[-1]]), ""
            return match, ""
        r = self.rng.random()  # "random"
        if r < 0.1 and cost > 0:
            return None, ""
        if escalation and r < 0.5:
            return self.rng.choice(list(escalation)), ""
        return match, ""


class ApiSeat:
    """A seat driven by any OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, model: str, url: str, api_key: str = "", temperature: float = 1.0, max_tokens: int = 200):
        self.model, self.url, self.api_key = model, url, api_key
        self.temperature, self.max_tokens = temperature, max_tokens

    def act(self, observation_text: str, hand: Hand) -> tuple[int | None, str]:
        import urllib.request

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": observation_text},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        request = urllib.request.Request(
            self.url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(request, timeout=240) as response:
            reply = json.load(response)["choices"][0]["message"]["content"]
        total, say, _ = parse_reply(reply, hand)
        return total, say


def make_seat(spec: str, key: str) -> tuple[object | None, str]:
    """Seat spec -> (seat object or None for human, display name)."""
    if spec == "human":
        return None, "you"
    if spec.startswith("api:"):
        model, _, url = spec[4:].rpartition("@")
        if not model or not url:
            raise SystemExit(f"bad api seat spec {spec!r}; use api:<model>@<base-url>")
        return ApiSeat(model, url, key or ""), model.split("/")[-1]
    style = spec.removeprefix("fish:")
    if style not in FISH_STYLES:
        raise SystemExit(f"unknown seat spec {spec!r}; use human, {', '.join(FISH_STYLES)}, or api:<model>@<url>")
    import zlib

    return FishSeat(style, seed=zlib.crc32(spec.encode()) % 1000), f"fish ({style})"


# ---------------------------------------------------------------------------
# Dealer: the game across hands — chips, button, score, and the chat thread
# ---------------------------------------------------------------------------


class Dealer:
    """Runs the game: owns the chips, rotates the button (by seat-flipping,
    two players), keeps the running score, and holds the chat thread — the
    append-only ground truth every view (browser, observation, log) slices."""

    def __init__(self, names: list[str], chips: int = 200, seed: int = 0):
        self.names = names
        self.chips = chips
        self.stacks = [chips, chips]  # physical player order, stable across hands
        self.totals = [0.0, 0.0]  # score in bb, physical order
        self.chat = []  # {"seat": physical | None, "who", "text", "kind"}
        self.seed = seed
        self.hand_no = 0
        self.flip = 0  # physical index of the button; hand seat = physical ^ flip
        self.hand: Hand | None = None

    def line(self, seat: int | None, text: str, kind: str):
        who = self.names[seat] if seat is not None else "dealer"
        self.chat.append({"seat": seat, "who": who, "text": text[:200], "kind": kind})
        del self.chat[:-400]

    def say(self, seat: int, text: str):
        if text:
            self.line(seat, text, "talk")

    def physical(self, hand_seat: int) -> int:
        return hand_seat ^ self.flip

    def hand_seat(self, physical: int) -> int:
        return physical ^ self.flip

    def new_hand(self) -> Hand:
        self.hand_no += 1
        self.flip = (self.hand_no - 1) % 2
        if min(self.stacks) == 0:  # busted: fresh stacks, the score keeps the truth
            self.stacks = [self.chips, self.chips]
            self.line(None, "— rebuy: fresh stacks —", "deal")
        self._before = self.stacks.copy()
        self.hand = Hand([self.stacks[self.flip], self.stacks[1 - self.flip]], seed=self.seed + self.hand_no)
        self.line(None, f"— hand {self.hand_no} —", "deal")
        if self.hand_no > 1:
            score = " · ".join(f"{self.names[i]} {self.totals[i]:+.1f} bb" for i in (0, 1))
            self.line(None, score, "score")
        return self.hand

    def collect(self) -> list[int]:
        """Fold the finished hand back into the game; returns chip deltas."""
        for hand_seat in (0, 1):
            self.stacks[self.physical(hand_seat)] = self.hand.stacks[hand_seat]
        deltas = [self.stacks[i] - self._before[i] for i in (0, 1)]
        for i in (0, 1):
            self.totals[i] += deltas[i] / BIG_BLIND
        return deltas


# ---------------------------------------------------------------------------
# Table: one browser session — seats, pacing, narration. No web imports.
# ---------------------------------------------------------------------------


class Table:
    """Glues seats to a Dealer and narrates the game into the chat thread.
    The game's pace belongs here: extra viewers' ticks are dropped."""

    def __init__(self, seats: list, names: list[str], chips: int = 200, seed: int = 0, talk: bool = True):
        self.seats = seats  # physical order; None = the human
        self.human = next((i for i, s in enumerate(seats) if s is None), None)
        self.watch = self.human is None
        self.talk = talk
        self.dealer = Dealer(names, chips, seed)
        self.lock = threading.Lock()
        self.last_tick = 0.0
        self._base_key = None
        self.dealer.new_hand()
        self._advance_ai()

    # -- driving the hand ----------------------------------------------------

    def _apply(self, physical: int, total: int | None, say: str):
        hand = self.dealer.hand
        if total is None:  # narrate with the word the number means
            word = "fold"
        else:
            word = next(
                (e["action"] for e in labeled_legal(hand)
                 if e["action"] != "fold" and action_to_total(hand, e["action"]) == total),
                f"raise {total}",
            )
        street_before = hand.street
        hand.step(total)
        if self.talk and say:
            self.dealer.say(physical, say)
        self.dealer.line(physical, word, "act")
        if hand.acting_seat is not None and hand.street > street_before:
            street = ["preflop", "flop", "turn", "river"][hand.street]
            self.dealer.line(None, f"{street}: {' '.join(hand.revealed())}", "deal")
        if hand.acting_seat is None:
            self._settle_narration()

    def _settle_narration(self):
        deltas = self.dealer.collect()
        winner = 0 if deltas[0] > 0 else 1
        hand = self.dealer.hand
        if deltas[0] == deltas[1] == 0:
            self.dealer.line(None, "Split pot.", "deal")
            return
        if any(hand.folded):
            how = f"{self.dealer.names[self.dealer.physical(hand.folded.index(True))]} folded"
        else:
            how = "showdown"
        self.dealer.line(None, f"{self.dealer.names[winner]} wins {deltas[winner]} chips ({how}).", "deal")

    def _advance_ai(self, limit: int = 50):
        """Let AI seats act until it's the human's turn, the hand ends, or
        (spectator view) one decision was made — so the page can animate."""
        for _ in range(limit):
            hand = self.dealer.hand
            if hand.acting_seat is None:
                return
            physical = self.dealer.physical(hand.acting_seat)
            seat = self.seats[physical]
            if seat is None:
                return
            obs = observation(hand, hand.acting_seat, self.dealer.names[physical], self.dealer.chat, self.talk)
            total, say = seat.act(obs, hand)
            self._apply(physical, total, say)
            if self.watch:
                return

    # -- the four things a browser can do -------------------------------------

    def act(self, action: str):
        with self.lock:
            hand = self.dealer.hand
            if hand.acting_seat is None or self.human is None:
                return
            if self.dealer.physical(hand.acting_seat) != self.human:
                return
            try:
                self._apply(self.human, action_to_total(hand, action), "")
            except ValueError:
                return  # stale or illegal request; the page will re-fetch the menu
            self._advance_ai()

    def post_chat(self, text: str):
        with self.lock:
            if self.human is not None and text.strip():
                self.dealer.say(self.human, text.strip())

    def tick(self):
        with self.lock:
            now = time.monotonic()
            if now - self.last_tick < 0.8:
                return
            self.last_tick = now
            if self.dealer.hand.acting_seat is not None:
                self._advance_ai()

    def next_hand(self):
        with self.lock:
            if self.dealer.hand.acting_seat is None:
                self.dealer.new_hand()
                self._advance_ai()

    # -- the JSON contract table.html speaks ----------------------------------

    def _street_base(self) -> list[int]:
        """bets snapshot at street start, for drawing carried-pot vs live bets."""
        hand = self.dealer.hand
        key = (self.dealer.hand_no, hand.street)
        if self._base_key != key:
            self._base_key = key
            self._base = hand.bets.copy()
        return self._base

    def state(self) -> dict:
        with self.lock:
            dealer, hand = self.dealer, self.dealer.hand
            done = hand.acting_seat is None
            base = [0] * 2 if done else self._street_base()
            show = [self.watch or done or physical == self.human for physical in (0, 1)]
            acting_physical = None if done else dealer.physical(hand.acting_seat)
            your_turn = self.human is not None and acting_physical == self.human
            return {
                "hand_no": dealer.hand_no,
                "watch": self.watch,
                "human": self.human,
                "names": dealer.names,
                "board": [card_ascii(c) for c in hand.revealed()],
                "pot": sum(base),  # carried pot only — live bets are drawn at the seats
                "stacks": [hand.stacks[dealer.hand_seat(p)] for p in (0, 1)],
                "bets": [hand.bets[dealer.hand_seat(p)] - base[dealer.hand_seat(p)] for p in (0, 1)],
                "hole": [
                    [card_ascii(c) for c in hand.holes[dealer.hand_seat(p)]] if show[p] else ["?", "?"]
                    for p in (0, 1)
                ],
                "button": dealer.flip,
                "to_act": acting_physical,
                "your_turn": your_turn,
                "legal": labeled_legal(hand) if your_turn else [],
                "done": done,
                "totals_bb": [round(t, 1) for t in dealer.totals],
                "chat": [{"who": e["who"], "text": e["text"], "kind": e["kind"]} for e in dealer.chat[-60:]],
                "talk": self.talk,
            }


# ---------------------------------------------------------------------------
# the bridge: everything below is plumbing — no poker. stdlib only.
# ---------------------------------------------------------------------------


def main():
    import argparse
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    parser = argparse.ArgumentParser(description="a poker table in the browser; any mix of players")
    parser.add_argument("--p0", default="human", help="seat 0: human, a fish style, or api:<model>@<url>")
    parser.add_argument("--p1", default="station", help="seat 1: same choices")
    parser.add_argument("--key0", default="", help="api key for seat 0 (api seats only)")
    parser.add_argument("--key1", default="", help="api key for seat 1")
    parser.add_argument("--chips", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-talk", action="store_true")
    parser.add_argument("--no-open", action="store_true", help="don't auto-open the browser")
    parser.add_argument("--port", type=int, default=8642)
    args = parser.parse_args()

    seat0, name0 = make_seat(args.p0, args.key0)
    seat1, name1 = make_seat(args.p1, args.key1)
    if seat0 is None and seat1 is None:
        raise SystemExit("two human seats need two browsers and a notion of identity — not built; keep one human")

    table = Table([seat0, seat1], [name0, name1], chips=args.chips, seed=args.seed, talk=not args.no_talk)
    page = Path(__file__).with_name("table.html")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _json(self, payload: dict):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/state":
                return self._json(table.state())
            body = page.read_bytes()  # read per request, so editing table.html only needs a refresh
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}") if length else {}
            if self.path == "/act":
                table.act(str(payload.get("action", "")))
            elif self.path == "/chat":
                table.post_chat(str(payload.get("text", "")))
            elif self.path == "/new":
                table.next_hand()
            elif self.path == "/tick":
                table.tick()
            self._json({"ok": True})

    url = f"http://localhost:{args.port}"
    print(f"table open at {url}  (ctrl-c to quit)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
