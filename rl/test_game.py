"""
Tests for rl/poker (rules, table, server NPCs) and rl/model_player (the LLM
client). Run: python -m rl.test_game
No test framework; each check prints a line and the script exits non-zero on
the first failure so it can gate a commit.
"""

import asyncio
import random

from rl.model_player import ModelPlayer
from rl.poker.game import (
    Game,
    Hand,
    INT_TO_CARD,
    card_ascii,
    get_5_score,
    get_7_score,
    labeled_legal,
)
from rl.poker.server import fish, fish_pick


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"{status} {name} {detail}")
    if not condition:
        raise SystemExit(1)


def acting_state(game: Game) -> dict:
    """The snapshot the acting seat's client would receive."""
    return game.state(game.physical(game.hand.acting_seat))


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
    game = Game(["a", "b"], seed=5)
    hand = game.hand
    menu = labeled_legal(hand)
    check("menu starts fold/call facing the blind", [m["action"] for m in menu[:2]] == ["fold", "call"])
    match, escalation = hand.legal_totals()
    for entry in menu:
        total = entry["total"]
        legal = total is None or total == match or total in escalation
        check(f"menu entry {entry['action']!r} carries a legal total", legal)
    state = acting_state(game)
    check("state prices the acting seat", state["match"] == match and state["escalation"] is not None)
    check("model: raise to N form", ModelPlayer.to_total(state, "raise to 12") == 12)
    check("model: allin", ModelPlayer.to_total(state, "allin") == state["escalation"][1])
    try:
        ModelPlayer.to_total(state, "quack")
        check("unknown action raises", False)
    except ValueError:
        check("unknown action raises", True)
    prompt = ModelPlayer.render_prompt(state)
    check("prompt shows ascii cards", "Your hole cards: " in prompt and "♥" not in prompt)
    check("prompt lists the menu", "Legal actions: " in prompt)


def test_fish():
    game = Game(["a", "b"], seed=21)
    state = acting_state(game)  # facing the blind: fold / call / raises all on the menu
    rng = random.Random(0)
    check("station calls", fish_pick(state, "station", rng) == state["match"])
    check("nit folds to a bet", fish_pick(state, "nit", rng) is None)
    maniac = fish_pick(state, "maniac", rng)
    check("maniac escalates", maniac is not None and maniac >= state["escalation"][0])
    lo, hi = state["escalation"]
    legal_random = all(
        (t is None or t == state["match"] or lo <= t <= hi)
        for t in (fish_pick(state, "random", rng) for _ in range(200))
    )
    check("random fish stays legal over 200 picks", legal_random)


def test_game_session():
    g = Game(["a", "b"], chips=200, seed=7)
    check("hand 1: physical 0 has the button", g.button == 0 and g.physical(0) == 0)
    while g.hand.acting_seat is not None:
        g.hand.step(g.hand.legal_totals()[0])
    g.collect()
    check("chips conserved through collect", sum(g.stacks) == 400, str(g.stacks))
    g.new_hand()
    check("hand 2: button rotated", g.button == 1 and g.physical(0) == 1)
    g.say(0, "hello")
    check("chat is ground truth on the game", g.chat[-1]["who"] == "a" and g.chat[-1]["kind"] == "talk")
    g.stacks = [400, 0]
    g.new_hand()
    check("bust triggers a rebuy", g.stacks[0] > 0 and g.stacks[1] > 0, str(g.stacks))

    g3 = Game(["a", "b", "c"], chips=100, seed=8)
    check("3-player geometry round-trips", all(g3.hand_seat(g3.physical(s)) == s for s in range(3)))
    while g3.hand.acting_seat is not None:
        g3.hand.step(g3.hand.legal_totals()[0])
    g3.collect()
    check("3-player chips conserved", sum(g3.stacks) == 300, str(g3.stacks))
    g3.new_hand()
    check("3-player button rotates", g3.button == 1)


def test_table_session():
    async def scenario():
        game = Game(["maniac", "station"], chips=200, seed=9, auto_deal=True, pause=0)
        tasks = [
            asyncio.create_task(game.run()),
            asyncio.create_task(fish(game, 0, "maniac", pace=0)),
            asyncio.create_task(fish(game, 1, "station", pace=0)),
        ]
        for _ in range(4000):
            if game.hand_no > 3:
                break
            await asyncio.sleep(0)
        for task in tasks:
            task.cancel()
        return game

    game = asyncio.run(scenario())
    check("fish clients play hands by themselves", game.hand_no > 3, f"hand_no={game.hand_no}")
    total = sum(game.hand.stacks) + sum(game.hand.bets)
    check("chips conserved in the live hand", total == sum(game.stacks) or total == 400, str(total))
    kinds = {e["kind"] for e in game.chat}
    check("narration present", {"deal", "act"} <= kinds, str(kinds))
    state = game.state(None)
    for field in ("board", "stacks", "bets", "hole", "legal", "chat", "pot", "to_act",
                  "done", "street", "blinds", "match", "escalation", "you"):
        check(f"state has {field!r}", field in state)
    check("spectator sees both holes", all(cards != ["?", "?"] for cards in state["hole"]))


def test_client_protocol():
    async def scenario():
        game = Game(["you", "fish (station)"], chips=200, seed=11)
        tasks = [
            asyncio.create_task(game.run()),
            asyncio.create_task(fish(game, 1, "station", pace=0)),
        ]
        await asyncio.sleep(0.02)
        check("clock parks awaiting seat 0", game.state(0)["your_turn"])
        state = game.state(0)
        for field in ("you", "hole", "board", "match", "escalation", "legal", "chat", "talk"):
            check(f"acting state has {field!r}", field in state)
        check("player stream hides opponent hole", game.state(0)["hole"][1] == ["?", "?"])
        check("spectator stream shows both", game.state(None)["hole"][1] != ["?", "?"])
        version_before = game.version
        game.give(0, state["match"])
        await asyncio.sleep(0.02)
        check("version bumps on action", game.version > version_before)
        check("play returned to seat 0", game.state(0)["your_turn"])
        version_watched = game.version
        waited = asyncio.create_task(game.wait_past(version_watched))
        game.give(0, 4)
        await asyncio.sleep(0.02)
        check("wait_past wakes on change", waited.done() and waited.result() > version_watched)
        check("raise applied, station answered", sum(game.hand.bets) >= 8, str(game.hand.bets))
        game.give(1, None)
        check("wrong-turn deposit dropped at the door", game._actions.empty())
        game.give(0, None)
        await asyncio.sleep(0.02)
        check("fold ends the hand", game.hand.acting_seat is None)
        game.request_new()
        await asyncio.sleep(0.02)
        check("next hand dealt on request", game.hand_no == 2)
        for task in tasks:
            task.cancel()

    asyncio.run(scenario())


def test_model_player_repair():
    def message(action=None, content=None):
        calls = []
        if action is not None:
            calls = [{"function": {"name": "act", "arguments": f'{{"action": "{action}", "say": "gl"}}'}}]
        return {"content": content, "tool_calls": calls}

    async def scenario(replies):
        player = ModelPlayer("test-model", "http://localhost:1")
        script = iter(replies)

        async def fake_call(messages):
            return next(script)

        player._call = fake_call
        state = Game(["m", "s"], seed=13).state(0)  # seat 0 to act, facing the blind
        return player, await player.decide(state), state

    player, (total, say), state = asyncio.run(scenario([message(action="call")]))
    check("model tool call -> total", total == state["match"] and say == "gl")
    check("clean call not billed invalid", player.invalid == 0)

    player, (total, _), state = asyncio.run(scenario([message(action="raise 3"), message(action="raise 6")]))
    check("illegal raise repaired by re-ask", total == 6)
    check("repair not billed invalid", player.invalid == 0)

    player, (total, _), state = asyncio.run(scenario([message(content="I fold I guess"), message(content="hmm")]))
    check("no tool call twice -> cheapest default", total is None)
    check("failure billed invalid", player.invalid == 1)

    player, (total, say), state = asyncio.run(
        scenario([message(content='{"action": "call", "say": "leaked"}')])
    )
    check("content-leaked JSON call repaired", total == state["match"] and say == "leaked")


if __name__ == "__main__":
    test_cards()
    test_scoring()
    test_hand_rules()
    test_hand_fuzz()
    test_vocabulary()
    test_fish()
    test_game_session()
    test_table_session()
    test_client_protocol()
    test_model_player_repair()
    print("all game tests passed")
