"""
Heads-up no-limit hold'em dealer. Standard library only.

One hand at a time, both players start every hand with a fresh stack, so a
"match" is just a sequence of independent hands whose chip deltas we add up.

Units: chips. Small blind 1, big blind 2, starting stack 200 (= 100 bb).
Results are reported in chips; divide by BB to get big blinds.

Two variance reducers are built in because they are what make a learning
curve readable at hobby sample sizes:
  * mirrored decks   -- every seed is played twice with the seats swapped
  * equity settlement -- when both players are all-in before the river the
                         pot is split by exact equity instead of one runout

Secrets (the opponent's hole cards) only ever live inside Dealer. Seats see
text observations, never the Dealer object.
"""

import itertools
import random
import re


SB = 1
BB = 2
STACK = 200
STREETS = ["preflop", "flop", "turn", "river"]
RANKS = "23456789TJQKA"
SUITS = "cdhs"


# ----------------------------------------------------------------------------
# Cards and hand evaluation
# ----------------------------------------------------------------------------


def card_str(card):
    return RANKS[card // 4] + SUITS[card % 4]


def cards_str(cards):
    return " ".join(card_str(c) for c in cards)


def parse_card(text):
    rank = RANKS.index(text[0].upper())
    suit = SUITS.index(text[1].lower())
    return rank * 4 + suit


def evaluate5(cards):
    """Score a 5-card hand. Higher is better. Ties score equal."""
    ranks = sorted((c // 4 for c in cards), reverse=True)
    suits = [c % 4 for c in cards]
    flush = len(set(suits)) == 1
    distinct = sorted(set(ranks), reverse=True)
    straight_high = None
    if len(distinct) == 5 and distinct[0] - distinct[4] == 4:
        straight_high = distinct[0]
    if distinct == [12, 3, 2, 1, 0]:
        straight_high = 3  # wheel: A-2-3-4-5 plays as five-high
    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    # Sort by (count, rank) so quads/trips/pairs come first, kickers after.
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = [count for _, count in groups]
    ordered = [rank for rank, _ in groups]
    if straight_high is not None and flush:
        category = 8
        ordered = [straight_high]
    elif shape[0] == 4:
        category = 7
    elif shape == [3, 2]:
        category = 6
    elif flush:
        category = 5
    elif straight_high is not None:
        category = 4
        ordered = [straight_high]
    elif shape[0] == 3:
        category = 3
    elif shape == [2, 2, 1]:
        category = 2
    elif shape[0] == 2:
        category = 1
    else:
        category = 0
    score = category
    for r in ordered:
        score = score * 13 + r
    # Pad so every category has the same number of digits.
    for _ in range(5 - len(ordered)):
        score = score * 13
    return score


def evaluate7(cards):
    """Best 5-card score out of 7 cards."""
    best = 0
    for combo in itertools.combinations(cards, 5):
        score = evaluate5(combo)
        if score > best:
            best = score
    return best


def equity(hole0, hole1, board, rng, samples=200):
    """P(seat 0 wins) + 0.5 P(tie), estimated by random runouts.

    Exact on the river (no runout needed). Monte Carlo earlier, which is
    plenty for a reward signal; the noise it adds is far below one hand's.
    """
    used = set(hole0) | set(hole1) | set(board)
    deck = [c for c in range(52) if c not in used]
    need = 5 - len(board)
    if need == 0:
        samples = 1
    total = 0.0
    for _ in range(samples):
        runout = rng.sample(deck, need) if need else []
        full = board + runout
        s0 = evaluate7(hole0 + full)
        s1 = evaluate7(hole1 + full)
        if s0 > s1:
            total += 1.0
        elif s0 == s1:
            total += 0.5
    return total / samples


# ----------------------------------------------------------------------------
# The hand
# ----------------------------------------------------------------------------


class Dealer:
    """One hand of heads-up NLHE.

    Seat `button` posts the small blind and acts first preflop, second after.
    The deck order is a pure function of `seed`, so two Dealers with the same
    seed and opposite buttons deal the same cards to opposite seats.
    """

    def __init__(self, seed, button=0, stack=STACK, talk=True, equity_samples=200):
        self.seed = seed
        self.button = button
        self.talk = talk
        self.equity_samples = equity_samples
        self.rng = random.Random(seed)
        deck = list(range(52))
        self.rng.shuffle(deck)
        self.hole = [deck[0:2], deck[2:4]]
        self.board_cards = deck[4:9]
        self.board_n = 0
        self.start_stack = stack
        self.stack = [stack, stack]
        self.bet = [0, 0]  # committed this street
        self.pot = 0  # committed on earlier streets
        self.street = 0
        self.min_raise = BB
        self.acted = [False, False]
        self.done = False
        self.winner = None  # None = showdown / equity; else the seat that won by fold
        self.delta = None
        self.history = []  # (street, seat, action text, say)
        self.invalid = [0, 0]
        self._post(self.button, SB)
        self._post(1 - self.button, BB)
        self.to_act = self.button

    # --- helpers ---

    def _post(self, seat, amount):
        amount = min(amount, self.stack[seat])
        self.stack[seat] -= amount
        self.bet[seat] += amount

    @property
    def board(self):
        return self.board_cards[: self.board_n]

    def to_call(self, seat):
        return min(max(self.bet) - self.bet[seat], self.stack[seat])

    def pot_total(self):
        return self.pot + self.bet[0] + self.bet[1]

    def all_in(self, seat):
        return self.stack[seat] == 0

    # --- legal actions ---

    def legal_actions(self, seat=None):
        """Action menu for the seat to act, as exact strings the seat must echo.

        Sizes are raise-to / bet-to totals for this street: the min-raise, then
        half pot, pot and two pots, then all-in. "bet" when nobody has bet yet,
        "raise" otherwise.
        """
        seat = self.to_act if seat is None else seat
        if self.done or seat != self.to_act:
            return []
        call = self.to_call(seat)
        actions = []
        if call > 0:
            actions.append("fold")
            actions.append("call")
        else:
            actions.append("check")
        if self.all_in(1 - seat):
            return actions  # nobody left to raise against
        max_to = self.bet[seat] + self.stack[seat]
        min_to = max(self.bet) + self.min_raise
        if min_to >= max_to:
            actions.append("allin")
            return actions
        pot_now = self.pot_total() + call
        verb = "bet" if max(self.bet) == 0 else "raise"
        sizes = [min_to]
        for frac in (0.5, 1.0, 2.0):
            sizes.append(max(self.bet) + int(round(frac * pot_now)))
        for to in sorted(set(sizes)):
            if min_to <= to < max_to:
                actions.append(f"{verb} {to}")
        actions.append("allin")
        return actions

    # --- applying an action ---

    def step(self, action, say=""):
        """Apply `action` for the seat to act. Raises ValueError if illegal."""
        seat = self.to_act
        legal = self.legal_actions(seat)
        action = action.strip().lower()
        if action not in legal:
            raise ValueError(f"illegal action {action!r}; legal: {legal}")
        if not self.talk:
            say = ""
        self.history.append((self.street, seat, action, say[:120]))
        opp = 1 - seat
        if action == "fold":
            self._finish(winner=opp)
            return
        if action == "check":
            pass
        elif action == "call":
            self._post(seat, self.to_call(seat))
        elif action == "allin":
            self._raise_to(seat, self.bet[seat] + self.stack[seat])
        else:
            to = int(action.split()[1])
            self._raise_to(seat, to)
        self.acted[seat] = True
        self._advance(seat)

    def _raise_to(self, seat, to):
        raise_size = to - max(self.bet)
        if raise_size >= self.min_raise:
            self.min_raise = raise_size
        self._post(seat, to - self.bet[seat])
        self.acted[1 - seat] = False  # the raise reopens action

    def _advance(self, seat):
        opp = 1 - seat
        settled = self.bet[0] == self.bet[1] or self.all_in(0) or self.all_in(1)
        both_acted = self.acted[0] and self.acted[1]
        # If the opponent is all-in and we just matched, or everyone acted
        # with equal bets, the street is over.
        if settled and (both_acted or self.all_in(opp) or (self.all_in(seat) and self.bet[seat] <= self.bet[opp])):
            self._next_street()
            return
        self.to_act = opp

    def _next_street(self):
        self.pot += self.bet[0] + self.bet[1]
        self.bet = [0, 0]
        self.acted = [False, False]
        self.min_raise = BB
        if self.all_in(0) or self.all_in(1):
            self._finish(winner=None)
            return
        if self.street == 3:
            self._finish(winner=None)
            return
        self.street += 1
        self.board_n = {1: 3, 2: 4, 3: 5}[self.street]
        self.to_act = 1 - self.button

    def _finish(self, winner):
        self.done = True
        self.to_act = None
        self.pot += self.bet[0] + self.bet[1]
        self.bet = [0, 0]
        invested = [self.start_stack - self.stack[0], self.start_stack - self.stack[1]]
        # Only the matched part of the pot is contested; the excess goes back.
        contested = 2 * min(invested)
        excess_seat = 0 if invested[0] > invested[1] else 1
        refund = invested[0] + invested[1] - contested
        if winner is not None:
            self.winner = winner
            share0 = contested if winner == 0 else 0.0
        else:
            rng = random.Random(self.seed + 7919)
            eq0 = equity(self.hole[0], self.hole[1], self.board, rng, self.equity_samples)
            share0 = contested * eq0
        won = [share0, contested - share0]
        won[excess_seat] += refund
        self.delta = [won[0] - invested[0], won[1] - invested[1]]

    def result(self):
        """Chip delta per seat (floats; equity settlement can be fractional)."""
        return self.delta

    # --- text observation ---

    def observation(self, seat):
        opp = 1 - seat
        role = "button (small blind)" if seat == self.button else "big blind"
        lines = []
        lines.append(f"Heads-up no-limit hold'em. Blinds {SB}/{BB}. 1 bb = {BB} chips. Starting stacks {STACK}.")
        lines.append(f"You are seat {seat}, the {role}.")
        lines.append(f"Your hole cards: {cards_str(self.hole[seat])}")
        board = cards_str(self.board) if self.board else "(none yet)"
        lines.append(f"Street: {STREETS[self.street]}. Board: {board}")
        lines.append(f"Pot: {self.pot_total()}. Your stack: {self.stack[seat]}. Opponent stack: {self.stack[opp]}.")
        lines.append("Action so far: " + self._history_text(seat))
        if self.done:
            lines.append(self._result_text(seat))
            return "\n".join(lines)
        call = self.to_call(seat)
        if call > 0:
            lines.append(f"To call: {call}.")
        lines.append("Legal actions: " + ", ".join(self.legal_actions(seat)))
        lines.append("Reply with exactly one legal action from the list.")
        if self.talk:
            lines.append('You may add table talk on a second line: say: <message>')
        return "\n".join(lines)

    def _history_text(self, seat):
        if not self.history:
            return "blinds posted."
        parts = []
        last_street = None
        for street, who, action, say in self.history:
            if street != last_street:
                parts.append(f"[{STREETS[street]}]")
                last_street = street
            name = "you" if who == seat else "opp"
            text = f"{name} {action}"
            if say:
                text += f' (says: "{say}")'
            parts.append(text)
        return " ".join(parts)

    def _result_text(self, seat):
        delta = self.delta[seat]
        opp = 1 - seat
        if self.winner is not None:
            how = "opponent folded" if self.winner == seat else "you folded"
        elif self.board_n == 5:
            how = f"showdown, opponent had {cards_str(self.hole[opp])}"
        else:
            how = f"all-in, settled by equity, opponent had {cards_str(self.hole[opp])}"
        verb = "won" if delta > 0 else "lost"
        return f"HAND OVER: you {verb} {abs(delta):.1f} chips ({how})."


# ----------------------------------------------------------------------------
# Seats: things that turn an observation into an action
# ----------------------------------------------------------------------------

ACTION_RE = re.compile(r"\b(fold|check|call|allin|raise\s+\d+|bet\s+\d+)\b", re.IGNORECASE)
SAY_RE = re.compile(r"say:\s*(.+)", re.IGNORECASE)


def parse_reply(text, legal):
    """Pull (action, say) out of free text. Returns (None, say) if no legal action found."""
    say = ""
    match = SAY_RE.search(text)
    if match:
        say = match.group(1).strip().strip('"')
        text = text[: match.start()]
    found = [m.group(1).lower() for m in ACTION_RE.finditer(text)]
    found = [re.sub(r"\s+", " ", f) for f in found]
    for action in reversed(found):  # the last action mentioned is the decision
        if action in legal:
            return action, say
    return None, say


def default_action(legal):
    """What an invalid reply turns into: the cheapest legal thing."""
    if "check" in legal:
        return "check"
    return "fold"


class FishSeat:
    """Scripted opponents. Exploitable on purpose; they are the beta meter."""

    def __init__(self, style="station", seed=0):
        self.style = style
        self.rng = random.Random(seed)
        self.invalid = 0

    def act(self, observation, legal):
        raises = [a for a in legal if a.startswith(("raise", "bet"))]
        if self.style == "station":  # calls everything, never raises
            return ("call" if "call" in legal else "check"), ""
        if self.style == "nit":  # folds to any bet, checks otherwise
            return ("fold" if "fold" in legal else "check"), ""
        if self.style == "maniac":  # raises when it can, else calls
            if raises:
                return self.rng.choice(raises), ""
            if "allin" in legal and self.rng.random() < 0.5:
                return "allin", ""
            return ("call" if "call" in legal else "check"), ""
        return self.rng.choice(legal), ""  # "random"


class PolicySeat:
    """Wraps any text -> text function (a language model) as a seat."""

    def __init__(self, generate):
        self.generate = generate
        self.invalid = 0
        self.decisions = 0

    def act(self, observation, legal):
        reply = self.generate(observation)
        action, say = parse_reply(reply, legal)
        self.decisions += 1
        if action is None:
            self.invalid += 1
            action = default_action(legal)
        return action, say


# ----------------------------------------------------------------------------
# Playing hands and matches
# ----------------------------------------------------------------------------


def play_hand(dealer, seats):
    """Run one hand to completion. `seats` is [seat0, seat1]. Returns chip deltas."""
    while not dealer.done:
        seat = dealer.to_act
        obs = dealer.observation(seat)
        legal = dealer.legal_actions(seat)
        action, say = seats[seat].act(obs, legal)
        dealer.step(action, say)
    return dealer.result()


def play_match(make_a, make_b, hands=100, seed=0, mirrored=True, talk=False, progress=None):
    """A vs B for `hands` hands. Returns A's bb/100 and both invalid rates.

    With `mirrored`, each seed is dealt twice with seats swapped, so card luck
    cancels exactly and only decisions remain. `hands` counts both copies.

    `progress(played, hands, total_chips)` is called after every hand; returning
    False stops the match early and the stats cover the hands actually played —
    so a budget cap or a kill signal never erases the data already paid for.
    """
    total = 0.0
    played = 0
    a = make_a()
    b = make_b()
    for i in range(hands):
        if mirrored:
            seed_i = seed + i // 2
            a_seat = i % 2  # second copy: A takes B's seat, same deck, same button
            button = (i // 2) % 2
        else:
            seed_i = seed + i
            a_seat = 0
            button = i % 2
        dealer = Dealer(seed_i, button=button, talk=talk)
        seats = [a, b] if a_seat == 0 else [b, a]
        deltas = play_hand(dealer, seats)
        total += deltas[a_seat]
        played += 1
        if progress and progress(played, hands, total) is False:
            break
    stats = {
        "hands": played,
        "bb_per_100": 100.0 * total / played / BB,
        "invalid_rate_a": getattr(a, "invalid", 0) / max(1, getattr(a, "decisions", played)),
        "invalid_rate_b": getattr(b, "invalid", 0) / max(1, getattr(b, "decisions", played)),
    }
    return stats
