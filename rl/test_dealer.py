"""
Scripted-hand tests for the dealer. Run: python -m rl.test_dealer
No test framework; each check prints a line and the script exits non-zero on
the first failure so it can gate a commit.
"""

import random

from rl.dealer import BB, Dealer, FishSeat, evaluate5, evaluate7, parse_card, parse_reply, play_hand, play_match


def cards(text):
    return [parse_card(t) for t in text.split()]


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"{status} {name} {detail}")
    if not condition:
        raise SystemExit(1)


def test_hand_rankings():
    royal = evaluate5(cards("As Ks Qs Js Ts"))
    quads = evaluate5(cards("9c 9d 9h 9s 2c"))
    full = evaluate5(cards("3c 3d 3h Kc Kd"))
    flush = evaluate5(cards("2h 7h 9h Jh Kh"))
    straight = evaluate5(cards("5c 6d 7h 8s 9c"))
    wheel = evaluate5(cards("Ac 2d 3h 4s 5c"))
    trips = evaluate5(cards("7c 7d 7h Ac 2d"))
    two_pair = evaluate5(cards("Ac Ad Kc Kd 2h"))
    pair = evaluate5(cards("Ac Ad 9c 5d 2h"))
    high = evaluate5(cards("Ac Kd 9c 5d 2h"))
    order = [royal, quads, full, flush, straight, wheel, trips, two_pair, pair, high]
    check("hand categories ordered", order == sorted(order, reverse=True))
    check("wheel below six-high straight", wheel < straight)
    check("kicker decides", evaluate5(cards("Ac Ad Kc 5d 2h")) > evaluate5(cards("As Ah Qc 5s 2d")))
    check("ties score equal", evaluate5(cards("Ac Kd 9c 5d 2h")) == evaluate5(cards("As Kh 9s 5h 2d")))
    seven = evaluate7(cards("As Ks Qs Js Ts 2c 3d"))
    check("evaluate7 finds the royal", seven == royal)


def test_fold_preflop():
    d = Dealer(seed=1, button=0)
    check("button acts first preflop", d.to_act == 0)
    check("button faces a bet", "fold" in d.legal_actions() and "check" not in d.legal_actions())
    d.step("fold")
    check("fold ends the hand", d.done)
    check("folding the small blind loses 1", d.result() == [-1, 1], str(d.result()))


def test_check_down_showdown():
    d = Dealer(seed=2, button=0)
    d.step("call")
    check("big blind acts after the limp", d.to_act == 1)
    d.step("check")
    check("flop dealt after preflop settles", d.street == 1 and len(d.board) == 3)
    check("big blind first postflop", d.to_act == 1)
    check("postflop menu says bet, not raise", "bet 2" in d.legal_actions() and not any(a.startswith("raise") for a in d.legal_actions()), str(d.legal_actions()))
    for _ in range(3):
        d.step("check")
        d.step("check")
    check("river check-down ends the hand", d.done)
    r = d.result()
    check("zero sum", abs(r[0] + r[1]) < 1e-9, str(r))
    check("pot of 4 chips changes hands", abs(r[0]) in (0.0, 2.0), str(r))


def test_raise_sizes_and_min_raise():
    d = Dealer(seed=3, button=0)
    legal = d.legal_actions()
    check("min-raise to 4 offered preflop", "raise 4" in legal, str(legal))
    check("all-in offered", "allin" in legal)
    d.step("raise 6")
    check("raise reopens action for the big blind", d.to_act == 1)
    legal = d.legal_actions(1)
    check("min re-raise respects last raise size", "raise 10" in legal and "raise 8" not in legal, str(legal))
    d.step("call")
    check("call after raise closes the street", d.street == 1)


def test_all_in_equity_settlement():
    d = Dealer(seed=4, button=0, equity_samples=400)
    d.hole = [cards("As Ah"), cards("Ks Kh")]
    d.step("allin")
    check("opponent faces the all-in", d.to_act == 1 and "call" in d.legal_actions())
    d.step("call")
    check("all-in call ends the hand", d.done)
    r = d.result()
    check("zero sum after equity settlement", abs(r[0] + r[1]) < 1e-9, str(r))
    # AA vs KK is ~82% for the aces: expected +0.64 * 200 = +128 chips.
    check("aces take roughly their equity", 105 < r[0] < 150, f"{r[0]:.1f}")


def test_illegal_action_rejected():
    d = Dealer(seed=5, button=0)
    try:
        d.step("check")
        check("illegal action raises", False)
    except ValueError:
        check("illegal action raises", True)


def test_parse_reply():
    legal = ["fold", "call", "raise 6", "raise 8", "allin"]
    check("bet parses", parse_reply("bet 4", ["check", "bet 4", "allin"]) == ("bet 4", ""))
    check("plain action parses", parse_reply("call", legal) == ("call", ""))
    check("last mentioned action wins", parse_reply("I could fold but I raise 6", legal)[0] == "raise 6")
    check("illegal size rejected", parse_reply("raise 7", legal)[0] is None)
    check("say parsed", parse_reply("raise 8\nsay: you got nothing", legal) == ("raise 8", "you got nothing"))
    check("garbage is None", parse_reply("hmm", legal)[0] is None)


def test_mirrored_symmetry():
    stats = play_match(lambda: FishSeat("station"), lambda: FishSeat("station"), hands=200, seed=10)
    check("identical deterministic players mirror to exactly 0", abs(stats["bb_per_100"]) < 1e-9, f"{stats['bb_per_100']:.3f}")
    check("no invalid actions from scripted seats", stats["invalid_rate_a"] == 0 and stats["invalid_rate_b"] == 0)


def test_random_play_is_zero_sum_and_terminates():
    for i in range(300):
        d = Dealer(seed=1000 + i, button=i % 2, equity_samples=20)
        r = play_hand(d, [FishSeat("random", seed=i), FishSeat("random", seed=-i)])
        check_ok = abs(r[0] + r[1]) < 1e-9 and d.done
        if not check_ok:
            check("random hands zero-sum and terminate", False, f"seed {1000 + i}: {r}")
    check("random hands zero-sum and terminate", True, "(300 hands)")


def test_maniac_beats_nit():
    stats = play_match(lambda: FishSeat("maniac", seed=1), lambda: FishSeat("nit", seed=2), hands=400, seed=20)
    check("maniac prints money against a nit", stats["bb_per_100"] > 20, f"{stats['bb_per_100']:.1f} bb/100")


if __name__ == "__main__":
    test_hand_rankings()
    test_fold_preflop()
    test_check_down_showdown()
    test_raise_sizes_and_min_raise()
    test_all_in_equity_settlement()
    test_illegal_action_rejected()
    test_parse_reply()
    test_mirrored_symmetry()
    test_random_play_is_zero_sum_and_terminates()
    test_maniac_beats_nit()
    print("all dealer tests passed")
