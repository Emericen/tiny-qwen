"""
Tests for rl/poker.py. Run: python test/test_poker.py
No test framework; each check prints a line and the script exits non-zero on
the first failure so it can gate a commit.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.poker import (
    Game,
    Hand,
    INT_TO_CARD,
    card_ascii,
    get_5_score,
    get_7_score,
    labeled_legal,
    render,
)


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"{status} {name} {detail}")
    if not condition:
        raise SystemExit(1)


def test_cards():
    check("52 distinct cards", len(set(INT_TO_CARD.values())) == 52)
    check("ascii round", card_ascii("♠A") == "As" and card_ascii("♥10") == "Th" and card_ascii("♣2") == "2c")


def test_scoring():
    def score(text):
        return get_5_score(text.split())

    royal = score("♠A ♠K ♠Q ♠J ♠10")
    wheel = score("♣A ♦2 ♥3 ♠4 ♣5")
    six_high = score("♣2 ♦3 ♥4 ♠5 ♣6")
    trips = score("♣7 ♦7 ♥7 ♣A ♦2")
    check("royal beats everything", royal > wheel > trips)
    check("wheel below six-high straight", wheel < six_high)
    check("ties score equal", score("♣A ♦K ♣9 ♦5 ♥2") == score("♠A ♥K ♠9 ♥5 ♦2"))
    seven = get_7_score("♠A ♠K ♠Q ♠J ♠10 ♣2 ♦3".split())
    check("get_7_score finds the royal", seven == royal)


def test_hand_rules():
    h = Hand([200, 200], seed=1)
    h.step(None)
    check("HU fold costs the blind", h.stacks == [199, 201], str(h.stacks))

    h = Hand([200, 200], seed=3)
    match, esc = h.legal_totals()
    check("preflop match is the big blind", match == 2)
    check("min raise floor is 4", esc.start == 4, str(esc))
    try:
        h.step(3)
        check("one-chip raise rejected", False)
    except ValueError:
        check("one-chip raise rejected", True)
    h.step(4)
    check("re-raise floor respects last size", h.legal_totals()[1].start == 6)

    h = Hand([100, 100, 100], seed=2)
    check("3-way: button opens preflop", h.acting_seat == 0)
    h.step(None)
    h.step(None)
    check("blinds fold -> BB scoops", h.stacks == [100, 99, 101], str(h.stacks))

    h = Hand([100, 100, 50], seed=3)
    h.holes = [["♠K", "♥K"], ["♠2", "♥3"], ["♠A", "♥A"]]
    h.board = ["♣4", "♦9", "♥J", "♣Q", "♦6"]
    h.step(100)
    h.step(h.legal_totals()[0])
    h.step(h.legal_totals()[0])
    check("side pot: AA wins main, KK wins side", h.stacks == [100, 0, 150], str(h.stacks))

    h = Hand([200, 200], seed=4)
    h.step(h.legal_totals()[0])
    check("BB option exists", h.acting_seat == 1 and h.street == 0)
    h.step(h.legal_totals()[0])
    check("street advances after option", h.street == 1)

    h = Hand([200, 200], button=1, seed=4)
    check("HU button=1: seat 1 posts small and opens", h.bets == [2, 1] and h.acting_seat == 1)

    h = Hand([100, 100, 100], button=1, seed=2)
    check("3-way button=1: blinds at 2 and 0, button opens", h.bets == [2, 0, 1] and h.acting_seat == 1)
    h.step(None)
    h.step(None)
    check("blinds fold -> BB scoops (rotated)", h.stacks == [101, 100, 99], str(h.stacks))


def test_hand_fuzz():
    rng = random.Random(0)
    for trial in range(20000):
        n = rng.randint(2, 6)
        stacks = [rng.randint(2, 300) for _ in range(n)]
        total0 = sum(stacks)
        h = Hand(stacks, button=rng.randrange(n), seed=trial)
        steps = 0
        while h.acting_seat is not None:
            match, esc = h.legal_totals()
            r = rng.random()
            h.step(None if r < 0.15 else (rng.choice(list(esc)) if esc and r < 0.55 else match))
            steps += 1
            assert steps < 200, "no termination"
            if h.acting_seat is not None:
                assert sum(h.stacks) + sum(h.bets) == total0, "conservation mid-hand"
        assert sum(h.stacks) == total0, "conservation at settle"
        assert min(h.stacks) >= 0
    check("fuzz: 20,000 hands (2-6 players, all buttons) conserve money and terminate", True)


def test_vocabulary():
    game = Game(["a", "b"], seed=5)
    hand = game.hand
    menu = labeled_legal(hand)
    check("menu starts fold/call facing the blind", [m["action"] for m in menu[:2]] == ["fold", "call"])
    match, escalation = hand.legal_totals()
    for entry in menu:
        total = entry["total"]
        legal = total is None or total == match or total in escalation
        check(f"menu entry {entry['action']!r} carries a legal total", legal)
    observation = game.observe(hand.acting_seat)
    check("observation prices the acting seat", observation["match"] == match and observation["escalation"])
    text = render(observation)
    check("render shows ascii cards", "Your hole cards: " in text and "♥" not in text)
    check("render lists totals, not verbs to parse", "Legal totals" in text)
    for entry in observation["legal"]:
        shown = "null" if entry["total"] is None else str(entry["total"])
        check(f"render prints the total for {entry['action']!r}", shown in text)
    check("render names the raise range", "to " + str(observation["escalation"][1]) in text)


def test_observe():
    game = Game(["a", "b", "c"], seed=6)
    acting = game.hand.acting_seat
    other = (acting + 1) % 3
    observation = game.observe(acting)
    for field in ("names", "you", "blinds", "street", "board", "hole", "stacks",
                  "bets", "pot", "match", "escalation", "legal", "chat", "talk"):
        check(f"observation has {field!r}", field in observation)
    check("observation holds only your own hole", len(observation["hole"]) == 2)
    bystander = game.observe(other)
    check("bystander gets no prices or menu", bystander["match"] is None and bystander["legal"] == [])
    game.say(0, "hi")
    check("talk reaches observations", game.observe(acting)["chat"][-1]["text"] == "hi")


def test_session():
    g = Game(["a", "b"], chips=200, seed=7)
    check("hand 1: seat 0 has the button", g.button == 0 and g.hand.button == 0)
    while g.hand.acting_seat is not None:
        g.act(g.hand.legal_totals()[0])  # everyone checks/calls to showdown
    check("chips conserved through the hand", sum(g.stacks) == 400, str(g.stacks))
    kinds = {e["kind"] for e in g.chat}
    check("narration present", {"deal", "act"} <= kinds, str(kinds))
    try:
        g.act(2)
        check("acting on a finished hand raises", False)
    except ValueError:
        check("acting on a finished hand raises", True)
    g.new_hand()
    check("hand 2: button rotated, blinds moved", g.button == 1 and g.hand.bets[1] == 1)
    bets_before = g.hand.bets.copy()
    try:
        g.act(3)  # below the min-raise floor
        check("illegal total raises", False)
    except ValueError:
        check("illegal total raises", True)
    check("illegal total leaves state intact", g.hand.bets == bets_before)
    g.stacks = [400, 0]
    g.new_hand()
    check("bust triggers a rebuy", g.stacks[0] > 0 and g.stacks[1] > 0, str(g.stacks))

    g3 = Game(["a", "b", "c"], chips=100, seed=8)
    while g3.hand.acting_seat is not None:
        g3.act(g3.hand.legal_totals()[0])
    check("3-player chips conserved", sum(g3.stacks) == 300, str(g3.stacks))
    g3.new_hand()
    check("3-player button rotates", g3.button == 1 and g3.hand.button == 1)
    for _ in range(20):  # a longer session settles and rotates without incident
        while g3.hand.acting_seat is not None:
            g3.act(g3.hand.legal_totals()[0])
        g3.new_hand()
    check("20-hand session conserves chips", sum(g3.stacks) == 300, str(g3.stacks))


if __name__ == "__main__":
    test_cards()
    test_scoring()
    test_hand_rules()
    test_hand_fuzz()
    test_vocabulary()
    test_observe()
    test_session()
    print("all poker tests passed")
