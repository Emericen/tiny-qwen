"""
Poker in three files:

    game.py      the rules and the state: cards, scoring, Hand (one episode),
                 Game (one table: chips, button, score, chat, the clock, and
                 every view of the state)
    server.py    a small FastAPI wrapper whose state is a map of Games — SSE
                 streams views out, POST /act {seat, total} brings actions in;
                 fish NPCs sit at the bottom as a convenience
    table.html   a frontend rendering what the server streams (2-player tables)

The game does not know who sits behind a seat. Humans, NPCs, and models are
all just clients that watch the stream and post a number when it is their
turn; decision policies live with the training code (rl/model_player.py),
not here. game.py imports nothing above the standard library.

The engine speaks ONE action: your cumulative chip total for the hand
(None = fold). check / call / bet / raise / all-in are presentation-layer
words for particular numbers — labeled_legal() hands every client the words
WITH their numbers, so nothing ever converts words back. Cards are emoji
strings ("♠A") internally and ASCII ("As") at the API boundary.

DEVIATION (documented): every all-in re-opens the action, even below a full
raise — real poker's incomplete-raise rule is dropped for legibility.
"""

import asyncio
import itertools
import random
from collections import Counter

# ---------------------------------------------------------------------------
# the deck
# ---------------------------------------------------------------------------

SMALL_BLIND, BIG_BLIND = 1, 2
CARDS_REVEALED = (0, 3, 4, 5)  # per street: preflop, flop, turn, river
STREETS = ("preflop", "flop", "turn", "river")
RANKS = "23456789TJQKA"
SUITS = "♣♦♥♠"

INT_TO_CARD = {c: SUITS[c % 4] + ("10" if c // 4 == 8 else RANKS[c // 4]) for c in range(52)}
CARD_TO_INT = {v: k for k, v in INT_TO_CARD.items()}
SUIT_ASCII = {"♣": "c", "♦": "d", "♥": "h", "♠": "s"}


def card_ascii(card: str) -> str:
    """'♠A' -> 'As', '♥10' -> 'Th' — the format the API boundary speaks."""
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
    ends when everyone still able to act has answered the current price.
    Seats are table order; `button` is just an index — where the blinds are
    posted and where each street's turn order starts."""

    def __init__(self, stacks: list[int], button: int = 0, seed: int | None = None):
        self.player_count = len(stacks)
        self.board, self.holes = self._deal(self.player_count, seed)
        self.stacks = stacks.copy()
        self.bets = [0] * self.player_count
        self.folded = [False] * self.player_count
        self.min_raise = BIG_BLIND
        self.street = 0
        self.button = button
        if self.player_count == 2:  # heads-up: the button posts the small blind
            small_blind_seat, big_blind_seat = button, (button + 1) % 2
        else:
            small_blind_seat = (button + 1) % self.player_count
            big_blind_seat = (button + 2) % self.player_count
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
        self.acting_seat = self._next_able_seat(self.button)  # first able seat after the button

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
# vocabulary: the acting seat's menu in poker words
# ---------------------------------------------------------------------------


def labeled_legal(hand: Hand) -> list[dict]:
    """The acting seat's menu: poker words for particular numbers. Every entry
    carries its number ("total", None = fold) — clients post that back
    verbatim, so nothing anywhere converts words to numbers. "n" is display
    detail: the call cost, or the raise total."""
    seat = hand.acting_seat
    match, escalation = hand.legal_totals()
    cost = match - hand.bets[seat]
    menu = []
    if cost > 0:
        menu.append({"action": "fold", "label": "Fold", "n": None, "total": None})
        menu.append({"action": "call", "label": "Call", "n": cost, "total": match})
    else:
        menu.append({"action": "check", "label": "Check", "n": None, "total": match})
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
                menu.append({"action": f"raise {n}", "label": label, "n": n, "total": n})
        menu.append({"action": "allin", "label": "All-in", "n": all_in, "total": all_in})
    return menu


# ---------------------------------------------------------------------------
# Game: one table — chips, button, score, chat, the clock, the views
# ---------------------------------------------------------------------------


class Game:
    """The game across hands at one table, any number of seats. Owns the
    chips, rotates the button, keeps the score and the chat thread, runs the
    clock, and serves every view of itself. It does NOT know who sits behind
    a seat: every action arrives from outside as give(seat, total), whether
    the sender is a browser, an NPC, or a model. Consumers learn of changes
    via a version counter: await wait_past(v)."""

    def __init__(self, names: list[str], chips: int = 200, seed: int = 0,
                 talk: bool = True, auto_deal: bool = False, pause: float = 2.4):
        self.names = list(names)
        self.n = len(self.names)
        self.talk = talk
        self.auto_deal = auto_deal  # next hand on a timer, vs on request_new()
        self.pause = pause  # seconds a finished hand stays up when auto-dealing
        self.chips = chips
        self.seed = seed
        self.stacks = [chips] * self.n
        self.totals = [0.0] * self.n  # score in bb
        self.chat = []  # {"seat": int | None (dealer), "who", "text", "kind"}
        self.hand_no = 0
        self.button = 0  # rotates every hand; Hand derives blinds and turn order from it
        self.hand: Hand | None = None
        self.version = 0
        self._changed = asyncio.Condition()
        self._actions: asyncio.Queue = asyncio.Queue()  # (seat, total) via give()
        self._new_request = asyncio.Event()
        self._base_key = None
        self.new_hand()

    # -- the chat thread: append-only ground truth ---------------------------

    def line(self, seat: int | None, text: str, kind: str):
        who = self.names[seat] if seat is not None else "dealer"
        self.chat.append({"seat": seat, "who": who, "text": text[:200], "kind": kind})
        del self.chat[:-400]

    def say(self, seat: int, text: str):
        if text:
            self.line(seat, text, "talk")

    # -- hands ---------------------------------------------------------------

    def new_hand(self) -> Hand:
        self.hand_no += 1
        self.button = (self.hand_no - 1) % self.n
        if min(self.stacks) == 0:  # busted players rebuy; the score keeps the truth
            self.stacks = [s if s > 0 else self.chips for s in self.stacks]
            self.line(None, "— rebuy: fresh stacks —", "deal")
        self._before = self.stacks.copy()
        self.hand = Hand(self.stacks, button=self.button, seed=self.seed + self.hand_no)
        self.line(None, f"— hand {self.hand_no} —", "deal")
        if self.hand_no > 1:
            score = " · ".join(f"{self.names[i]} {self.totals[i]:+.1f} bb" for i in range(self.n))
            self.line(None, score, "score")
        return self.hand

    def collect(self) -> list[int]:
        """Fold the finished hand back into the table; returns chip deltas."""
        self.stacks = self.hand.stacks.copy()
        deltas = [self.stacks[i] - self._before[i] for i in range(self.n)]
        for i in range(self.n):
            self.totals[i] += deltas[i] / BIG_BLIND
        return deltas

    # -- change signal: no registry, just a version --------------------------

    async def _notify(self):
        async with self._changed:
            self.version += 1
            self._changed.notify_all()

    async def wait_past(self, version: int) -> int:
        """Park until the table has changed past `version`; return the new one."""
        async with self._changed:
            await self._changed.wait_for(lambda: self.version > version)
            return self.version

    # -- inputs from the server ----------------------------------------------

    def give(self, seat: int, total: int | None):
        """An action arrives for `seat`. Accepted only when it is that seat's
        turn — anything else is a stale or confused client and is dropped."""
        if self.hand.acting_seat == seat:
            self._actions.put_nowait((seat, total))

    async def post_chat(self, seat: int, text: str):
        if 0 <= seat < self.n and text.strip():
            self.say(seat, text.strip())
            await self._notify()

    def request_new(self):
        self._new_request.set()

    # -- the clock -----------------------------------------------------------

    async def run(self):
        """The table runs itself: park until the acting seat's number arrives,
        apply it, broadcast. Who produced the number is not the game's business."""
        await self._notify()
        while True:
            hand = self.hand
            if hand.acting_seat is None:  # hand over: deal the next one
                if self.auto_deal:
                    await asyncio.sleep(self.pause)
                else:
                    await self._new_request.wait()
                    self._new_request.clear()
                self.new_hand()
                await self._notify()
                continue
            seat, total = await self._actions.get()
            if seat != hand.acting_seat:
                continue  # queued before the turn moved on
            try:
                self._apply(seat, total)
            except ValueError:
                continue  # an illegal number from a stale view; keep waiting
            await self._notify()

    def _apply(self, seat: int, total: int | None):
        hand = self.hand
        match, escalation = hand.legal_totals()
        cost = match - hand.bets[hand.acting_seat]
        if total is None:  # narrate with the word the number means
            word = "fold"
        elif total == match:
            word = "call" if cost > 0 else "check"
        elif escalation and total == escalation[-1]:
            word = "allin"
        else:
            word = f"raise {total}"
        street_before = hand.street
        hand.step(total)
        self.line(seat, word, "act")
        if hand.acting_seat is not None and hand.street > street_before:
            self.line(None, f"{STREETS[hand.street]}: {' '.join(hand.revealed())}", "deal")
        if hand.acting_seat is None:
            self._settle_narration()

    def _settle_narration(self):
        deltas = self.collect()
        if all(d == 0 for d in deltas):
            self.line(None, "Split pot.", "deal")
            return
        how = "showdown" if self.hand.folded.count(False) > 1 else "everyone else folded"
        winners = ", ".join(f"{self.names[i]} +{d}" for i, d in enumerate(deltas) if d > 0)
        self.line(None, f"{winners} chips ({how}).", "deal")

    # -- views ---------------------------------------------------------------

    def _street_base(self) -> list[int]:
        """bets snapshot at street start, for drawing carried-pot vs live bets."""
        key = (self.hand_no, self.hand.street)
        if self._base_key != key:
            self._base_key = key
            self._base = self.hand.bets.copy()
        return self._base

    def state(self, for_seat: int | None) -> dict:
        """One consumer's snapshot. for_seat None = spectator (v1: sees all
        holes); a seat number sees its own cards until showdown, plus its menu
        and prices when it is that seat's turn."""
        hand = self.hand
        done = hand.acting_seat is None
        base = [0] * self.n if done else self._street_base()
        show = [for_seat is None or done or p == for_seat for p in range(self.n)]
        your_turn = for_seat is not None and hand.acting_seat == for_seat
        match, escalation = hand.legal_totals() if your_turn else (None, range(0))
        return {
            "hand_no": self.hand_no,
            "watch": for_seat is None,
            "you": for_seat,
            "names": self.names,
            "blinds": [SMALL_BLIND, BIG_BLIND],
            "street": hand.street,
            "board": [card_ascii(c) for c in hand.revealed()],
            "pot": sum(base),  # carried pot only — live bets are drawn at the seats
            "stacks": list(hand.stacks),
            "bets": [hand.bets[p] - base[p] for p in range(self.n)],
            "hole": [
                [card_ascii(c) for c in hand.holes[p]] if show[p] else ["?", "?"]
                for p in range(self.n)
            ],
            "button": self.button,
            "to_act": hand.acting_seat,
            "your_turn": your_turn,
            "match": match,
            "escalation": [escalation.start, escalation[-1]] if escalation else None,
            "legal": labeled_legal(hand) if your_turn else [],
            "done": done,
            "totals_bb": [round(t, 1) for t in self.totals],
            "chat": [{"who": e["who"], "text": e["text"], "kind": e["kind"]} for e in self.chat[-60:]],
            "talk": self.talk,
        }
