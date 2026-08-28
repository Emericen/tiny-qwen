"""
Poker in four nouns:

    game.py      THE GAME:    state per the rules — pure library, no entry point
    seat.py      THE PLAYERS: one Seat class (human, fish, model) answering views
    server.py    THE DOOR:    FastAPI + SSE — streams views out, takes actions in
    table.html   THE GLASS:   renders what the door streams

This file is data structures per the game's rules and nothing else: cards and
scoring, one Hand (an episode), the Dealer (the game across hands: chips,
button, score, chat), and the Table — the async loop that assembles per-seat
views, awaits whoever's turn it is, and notifies subscribers on every event.
It imports nothing above it and knows nothing of HTTP, models, or pixels.

The engine speaks ONE action: your cumulative chip total for the hand
(None = fold). check / call / bet / raise / all-in are presentation-layer
words for particular numbers. Cards are emoji strings ("♠A") for humans and
converted to ASCII ("As") at the model/browser boundary.

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
# vocabulary: the acting seat's menu in poker words
# ---------------------------------------------------------------------------


def labeled_legal(hand: Hand) -> list[dict]:
    """The acting seat's menu as poker words. The browser buttons and the
    model prompt both consume this, so the two can never disagree."""
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


# ---------------------------------------------------------------------------
# Dealer: the game across hands — chips, button, score, and the chat thread
# ---------------------------------------------------------------------------


class Dealer:
    """Owns the chips, rotates the button (by seat-flipping, two players),
    keeps the running score, and holds the chat thread — the append-only
    ground truth every view (browser, prompt, log) slices."""

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
# Table: the loop — views out, answers in, subscribers notified. Async, pure.
# ---------------------------------------------------------------------------


class Table:
    """Assembles per-seat views, awaits whoever's turn it is (the Seat
    protocol: await act(view) -> (commit_to, say)), applies answers, narrates
    into the chat thread, and pings subscriber queues on every event. The
    game's clock lives here — the display just watches."""

    def __init__(self, seats: list, names: list[str] | None = None, chips: int = 200,
                 seed: int = 0, talk: bool = True, pace: float = 0.8):
        self.seats = seats  # physical order; protocol objects (see seat.py)
        names = names or [getattr(s, "name", f"seat {i}") for i, s in enumerate(seats)]
        self.human = next((i for i, s in enumerate(seats) if getattr(s, "is_human", False)), None)
        self.watch = self.human is None
        self.talk = talk
        self.pace = pace
        self.dealer = Dealer(names, chips, seed)
        self.subscribers: set[asyncio.Queue] = set()
        self._new_request = asyncio.Event()
        self._base_key = None
        self.dealer.new_hand()

    # -- subscriptions (the door hangs its streams here) ---------------------

    def subscribe(self) -> asyncio.Queue:
        queue = asyncio.Queue()
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue):
        self.subscribers.discard(queue)

    def _notify(self):
        for queue in list(self.subscribers):
            queue.put_nowait(True)

    # -- inputs from the door ------------------------------------------------

    def give(self, action: str):
        """Route the human's click to their seat — only when it's their turn."""
        hand = self.dealer.hand
        if self.human is None or hand.acting_seat is None:
            return
        if self.dealer.physical(hand.acting_seat) == self.human:
            self.seats[self.human].give(action)

    def post_chat(self, text: str):
        if self.human is not None and text.strip():
            self.dealer.say(self.human, text.strip())
            self._notify()

    def request_new(self):
        self._new_request.set()

    # -- the clock -----------------------------------------------------------

    async def run(self):
        """The game runs itself; viewers merely render what they're told."""
        self._notify()
        while True:
            hand = self.dealer.hand
            if hand.acting_seat is None:  # hand over: deal the next one
                if self.watch:
                    await asyncio.sleep(self.pace * 3)
                else:
                    await self._new_request.wait()
                    self._new_request.clear()
                self.dealer.new_hand()
                self._notify()
                continue
            physical = self.dealer.physical(hand.acting_seat)
            seat = self.seats[physical]
            total, say = await seat.act(self._view(physical))
            try:
                self._apply(physical, total, say)
            except ValueError:
                if getattr(seat, "is_human", False):
                    continue  # stale or garbage click; re-await a fresh one
                raise
            self._notify()
            if not getattr(seat, "is_human", False) and self.pace:
                await asyncio.sleep(self.pace)

    # -- driving the hand ----------------------------------------------------

    def _apply(self, physical: int, total: int | None, say: str):
        hand = self.dealer.hand
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

    # -- views ---------------------------------------------------------------

    def _view(self, physical: int) -> dict:
        """What the acting seat may know. Built only for the seat to act —
        match/escalation/menu are that seat's numbers."""
        hand = self.dealer.hand
        seat = self.dealer.hand_seat(physical)
        match, escalation = hand.legal_totals()
        return {
            "seat": seat,
            "name": self.dealer.names[physical],
            "player_count": hand.player_count,
            "street": hand.street,
            "board": hand.revealed(),
            "hole": hand.holes[seat],
            "stacks": hand.stacks,
            "bets": hand.bets,
            "pot": sum(hand.bets),
            "price": max(hand.bets),
            "match": match,
            "cost": match - hand.bets[seat],
            "escalation": [escalation.start, escalation[-1]] if escalation else None,
            "menu": labeled_legal(hand),
            "chat": [
                {"who": e["who"], "text": e["text"]}
                for e in self.dealer.chat
                if e["kind"] == "talk"
            ][-8:],
            "talk": self.talk,
        }

    # -- the JSON contract table.html speaks ---------------------------------

    def _street_base(self) -> list[int]:
        """bets snapshot at street start, for drawing carried-pot vs live bets."""
        hand = self.dealer.hand
        key = (self.dealer.hand_no, hand.street)
        if self._base_key != key:
            self._base_key = key
            self._base = hand.bets.copy()
        return self._base

    def state(self, for_seat: int | None) -> dict:
        """One subscriber's snapshot. for_seat None = spectator (v1: sees all
        holes); a seat number sees its own cards until showdown."""
        dealer, hand = self.dealer, self.dealer.hand
        done = hand.acting_seat is None
        base = [0] * 2 if done else self._street_base()
        show = [for_seat is None or done or physical == for_seat for physical in (0, 1)]
        acting_physical = None if done else dealer.physical(hand.acting_seat)
        your_turn = for_seat is not None and acting_physical == for_seat
        return {
            "hand_no": dealer.hand_no,
            "watch": for_seat is None,
            "human": for_seat,
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
