"""
Tests for rl/game/game.py. Run: python -m rl.test_game
No test framework; each check prints a line and the script exits non-zero on
the first failure so it can gate a commit.
"""

import random

from rl.poker.game import (
    Dealer,
    FishSeat,
    Hand,
    HumanSeat,
    INT_TO_CARD,
    NOT_READY,
    Table,
    action_to_total,
    card_ascii,
    get_5_score,
    get_7_score,
    labeled_legal,
)
from rl.poker.models import parse_reply, render_prompt


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


def test_hand_fuzz():
    rng = random.Random(0)
    for trial in range(20000):
        n = rng.randint(2, 6)
        stacks = [rng.randint(2, 300) for _ in range(n)]
        total0 = sum(stacks)
        h = Hand(stacks, seed=trial)
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
    check("fuzz: 20,000 hands (2-6 players) conserve money and terminate", True)


def test_vocabulary():
    h = Hand([200, 200], seed=5)
    menu = labeled_legal(h)
    check("menu starts fold/call facing the blind", [m["action"] for m in menu[:2]] == ["fold", "call"])
    for entry in menu:
        total = action_to_total(h, entry["action"])
        legal = total is None or total == h.legal_totals()[0] or total in h.legal_totals()[1]
        check(f"menu action {entry['action']!r} round-trips", legal)
    m, esc = h.legal_totals()
    cost = m - h.bets[h.acting_seat]
    check("parse: last action wins", parse_reply("I could fold but raise 6", m, esc, cost)[0] == 6)
    check("parse reports validity", parse_reply("call", m, esc, cost)[2] and not parse_reply("hmm", m, esc, cost)[2])
    check("parse: call", parse_reply("call", m, esc, cost)[0] == m)
    check("parse: allin", parse_reply("allin!", m, esc, cost)[0] == esc[-1])
    check("parse: pair form of escalation", parse_reply("raise 6", m, [esc.start, esc[-1]], cost)[0] == 6)
    check("parse: say extracted", parse_reply("call\nsay: nice hand", m, esc, cost)[:2] == (m, "nice hand"))
    check("parse: garbage folds facing a bet", parse_reply("hmm", m, esc, cost)[0] is None)


def test_dealer_session():
    d = Dealer(["a", "b"], chips=200, seed=7)
    d.new_hand()
    check("hand 1: physical 0 has the button", d.flip == 0 and d.physical(0) == 0)
    while d.hand.acting_seat is not None:
        d.hand.step(d.hand.legal_totals()[0])
    d.collect()
    check("chips conserved through collect", sum(d.stacks) == 400, str(d.stacks))
    d.new_hand()
    check("hand 2: button rotated", d.flip == 1 and d.physical(0) == 1)
    d.say(0, "hello")
    check("chat is ground truth on the dealer", d.chat[-1]["who"] == "a" and d.chat[-1]["kind"] == "talk")
    d.stacks = [400, 0]
    d.new_hand()
    check("bust triggers a rebuy", d.stacks[0] > 0 and d.stacks[1] > 0 and d.chat[-3]["text"].startswith("— rebuy"), str(d.stacks))


def test_table_session():
    table = Table([FishSeat("maniac", 1), FishSeat("station", 2)], ["m", "s"], chips=200, seed=9, talk=True)
    for _ in range(400):  # ticks: maniac vs station, several hands
        table.last_tick = 0.0
        table.tick()
        if table.dealer.hand.acting_seat is None:
            table.next_hand()
        if table.dealer.hand_no > 3:
            break
    check("watch mode plays multiple hands", table.dealer.hand_no > 3, f"hand_no={table.dealer.hand_no}")
    check("chips conserved in the live hand", sum(table.dealer.hand.stacks) + sum(table.dealer.hand.bets) == 400)
    kinds = {e["kind"] for e in table.dealer.chat}
    check("narration present", {"deal", "act"} <= kinds, str(kinds))
    state = table.state()
    for field in ("board", "stacks", "bets", "hole", "legal", "chat", "pot", "to_act", "done"):
        check(f"state has {field!r}", field in state)
    check("spectator sees both holes", all(cards != ["?", "?"] for cards in state["hole"]))


def test_seat_protocol():
    table = Table([HumanSeat(), FishSeat("station", 2)], ["you", "s"], chips=200, seed=11)
    check("loop pauses on the human (NOT_READY)", table.state()["your_turn"])
    view = table._view(0)
    for field in ("seat", "hole", "board", "match", "cost", "escalation", "menu", "chat", "talk"):
        check(f"view has {field!r}", field in view)
    prompt = render_prompt(view)
    check("prompt shows ascii hole cards", "Your hole cards: " in prompt and "♥" not in prompt)
    check("prompt lists the menu", "Legal actions: " in prompt)
    table.act("call")
    check("human call advances into the hand", table.dealer.hand.street >= 0 and table.state()["your_turn"])
    table.act("raise 4")
    state = table.state()
    check("human raise applied, station answered", sum(table.dealer.hand.bets) >= 8, str(table.dealer.hand.bets))
    table.act("fold")
    check("human fold ends the hand", table.dealer.hand.acting_seat is None)
    check("seat protocol NOT_READY sentinel exported", NOT_READY is not None)


if __name__ == "__main__":
    test_cards()
    test_scoring()
    test_hand_rules()
    test_hand_fuzz()
    test_vocabulary()
    test_dealer_session()
    test_table_session()
    test_seat_protocol()
    print("all game tests passed")
