"""
A poker game server. The UI lives in table.html; the rules live in dealer.py.

Any mix of players can sit at the two seats:

    python -m rl.game                                      # you vs a calling station
    python -m rl.game --p1 maniac                          # you vs another fish
    python -m rl.game --p1 api:kimi-k3@https://api.fireworks.ai/inference/v1 --key1 $KEY
    python -m rl.game --p0 api:grok-4.5@https://api.x.ai/v1 --key0 $XAI_API_KEY --p1 station

Seat specs: "human", a fish style (station | nit | maniac | random), or
"api:<model>@<openai-compatible base url>". With no human seat the page
becomes a spectator view: both hands face up, one AI decision per tick.

Then open http://localhost:8642. Any number of browsers can watch the same
table; the server paces the game, so extra viewers don't speed it up.
"""

import argparse
import json
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rl.game.dealer import BB, Dealer, FishSeat, PolicySeat, cards_str

FISH_STYLES = ("station", "nit", "maniac", "random")

SYSTEM = (
    "You are playing heads-up no-limit Texas hold'em for chips. "
    "Read the state and reply with exactly one legal action copied from the list. "
    "You may add one more line starting with 'say:' to talk to your opponent."
)


def api_seat(model, url, api_key, temperature=1.0, max_tokens=200):
    """A seat driven by any OpenAI-compatible /chat/completions endpoint."""

    def generate(observation):
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": observation},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        request = urllib.request.Request(
            url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.load(response)
        return data["choices"][0]["message"]["content"]

    return PolicySeat(generate)


def make_seat(spec, key):
    """Seat spec -> (seat object or None for human, display name)."""
    if spec == "human":
        return None, "you"
    if spec.startswith("api:"):
        model, _, url = spec[4:].rpartition("@")
        if not model or not url:
            raise SystemExit(f"bad api seat spec {spec!r}; use api:<model>@<base-url>")
        return api_seat(model, url, key or ""), model.split("/")[-1]
    style = spec.removeprefix("fish:")
    if style not in FISH_STYLES:
        raise SystemExit(f"unknown seat spec {spec!r}; use human, {', '.join(FISH_STYLES)}, or api:<model>@<url>")
    return FishSeat(style, seed=hash(spec) % 1000), f"fish ({style})"


class Table:
    """One long session of hands between two seats (human seats are None)."""

    def __init__(self, seats, names, talk):
        self.seats = seats
        self.names = names
        self.human = next((i for i, s in enumerate(seats) if s is None), None)
        self.watch = self.human is None
        self.talk = talk
        self.hand_no = 0
        self.totals = [0.0, 0.0]
        self.chat = []  # the table-talk thread: {"seat", "who", "text", "kind"}, across hands
        self.last_tick = 0.0
        self.lock = threading.Lock()
        self._deal()

    def _deal(self):
        self.hand_no += 1
        self.settled = False
        # talk=False: says never enter the dealer's history — the chat thread is the one channel
        self.dealer = Dealer(seed=self.hand_no, button=self.hand_no % 2, talk=False)
        self._line(None, f"— hand {self.hand_no} —", "deal")
        if self.hand_no > 1:
            score = " · ".join(
                f"{self.names[i]} {round(t / BB, 1):+g} bb" for i, t in enumerate(self.totals)
            )
            self._line(None, score, "score")
        self._advance_ai()

    def _line(self, seat, text, kind):
        self.chat.append({"seat": seat, "who": self.names[seat] if seat is not None else "dealer",
                          "text": text[:200], "kind": kind})
        del self.chat[:-200]

    def _say(self, seat, text):
        if self.talk and text:
            self._line(seat, text, "talk")

    def _step(self, seat, action):
        """Apply an action, narrating street reveals and results into the thread."""
        before = self.dealer.board_n
        self.dealer.step(action)
        self._line(seat, action, "act")
        if self.dealer.board_n > before:
            street = {3: "flop", 4: "turn", 5: "river"}[self.dealer.board_n]
            self._line(None, f"{street}: {self._pretty(cards_str(self.dealer.board))}", "deal")
        if self.dealer.done:
            self._settle()

    @staticmethod
    def _pretty(cards):
        return cards.translate(str.maketrans("shdc", "♠♥♦♣"))

    def _observe(self, seat):
        """The dealer's observation, plus the recent table talk for this seat."""
        obs = self.dealer.observation(seat)
        if not self.talk:
            return obs
        lines = [
            f'{"you" if entry["seat"] == seat else "opponent"}: {entry["text"]}'
            for entry in self.chat if entry["kind"] == "talk"
        ][-8:]
        if lines:
            obs += "\nTable talk so far:\n" + "\n".join(lines)
        return obs + '\nYou may add table talk on a second line: say: <message>'

    def _settle(self):
        """Add the finished hand to the totals and the thread, exactly once."""
        if self.dealer.done and not self.settled:
            self.settled = True
            self.totals[0] += self.dealer.result()[0]
            self.totals[1] += self.dealer.result()[1]
            self._line(None, self._result_summary(), "deal")

    def _result_summary(self):
        d = self.dealer
        if self.human is not None:
            return d._result_text(self.human)
        delta = d.result()
        w = 0 if delta[0] >= 0 else 1
        if d.winner is not None:
            how = f"{self.names[1 - d.winner]} folded"
        elif d.board_n == 5:
            how = "showdown"
        else:
            how = "all-in, settled by equity"
        if abs(delta[w]) < 1e-9:
            return f"Split pot ({how})."
        return f"{self.names[w]} wins {abs(delta[w]):.1f} chips ({how})."

    def _advance_ai(self, limit=50):
        """Let AI seats act until it's the human's turn, the hand ends, or
        (spectator view) one decision was made — so the page can animate."""
        for _ in range(limit):
            if self.dealer.done:
                self._settle()
                return
            seat = self.dealer.to_act
            if self.seats[seat] is None:
                return
            action, say = self.seats[seat].act(self._observe(seat), self.dealer.legal_actions(seat))
            self._say(seat, say)
            self._step(seat, action)
            if self.dealer.done:
                self._settle()
                return
            if self.watch:
                return

    def act(self, action):
        with self.lock:
            if self.dealer.done or self.human is None or self.dealer.to_act != self.human:
                return
            try:
                self._step(self.human, action)
            except ValueError:
                return  # stale or illegal request; the page will re-fetch the menu
            self._advance_ai()

    def post_chat(self, text):
        with self.lock:
            if self.human is not None:
                self._say(self.human, text.strip())

    def tick(self):
        with self.lock:
            # the game's pace belongs to the server: extra viewers' ticks are dropped
            now = time.monotonic()
            if now - self.last_tick < 0.8:
                return
            self.last_tick = now
            if not self.dealer.done:
                self._advance_ai()

    def next_hand(self):
        with self.lock:
            if self.dealer.done:
                self._deal()

    def _labeled_legal(self):
        """The dealer's menu, with a semantic label per action for the buttons."""
        d = self.dealer
        seat = d.to_act
        call = d.to_call(seat)
        pot_now = d.pot_total() + call
        min_to = max(d.bet) + d.min_raise
        sizes = {
            min_to: "Min",
            max(d.bet) + int(round(0.5 * pot_now)): "½ pot",
            max(d.bet) + int(round(1.0 * pot_now)): "Pot",
            max(d.bet) + int(round(2.0 * pot_now)): "2× pot",
        }
        labeled = []
        for action in d.legal_actions(seat):
            if action == "fold":
                labeled.append({"action": action, "label": "Fold", "n": None})
            elif action == "check":
                labeled.append({"action": action, "label": "Check", "n": None})
            elif action == "call":
                labeled.append({"action": action, "label": "Call", "n": call})
            elif action == "allin":
                labeled.append({"action": action, "label": "All-in", "n": None})
            else:
                to = int(action.split()[1])
                labeled.append({"action": action, "label": sizes.get(to, action), "n": to})
        return labeled

    def state(self):
        d = self.dealer
        show = [self.watch or d.done or i == self.human for i in range(2)]
        your_turn = self.human is not None and not d.done and d.to_act == self.human
        state = {
            "hand_no": self.hand_no,
            "watch": self.watch,
            "human": self.human,
            "names": self.names,
            "board": cards_str(d.board).split() if d.board else [],
            "pot": d.pot,  # carried pot only — live bets are drawn in front of the seats
            "stacks": d.stack,
            "bets": d.bet,
            "hole": [cards_str(d.hole[i]).split() if show[i] else ["?", "?"] for i in range(2)],
            "button": d.button,
            "to_act": d.to_act,
            "your_turn": your_turn,
            "legal": self._labeled_legal() if your_turn else [],
            "done": d.done,
            "totals_bb": [round(t / BB, 1) for t in self.totals],
            "chat": [{"who": e["who"], "text": e["text"], "kind": e["kind"]} for e in self.chat[-60:]],
            "talk": self.talk,
        }
        if d.done:
            state["result_text"] = self._result_summary()
        return state


# ---------------------------------------------------------------- the app

app = FastAPI()
table: Table = None  # set by main()
PAGE = Path(__file__).parent / "table.html"


class Act(BaseModel):
    action: str


class Chat(BaseModel):
    text: str


@app.get("/")
def page():
    return FileResponse(PAGE)  # read per request, so editing table.html only needs a refresh


@app.get("/state")
def state():
    return table.state()


@app.post("/act")
def act(body: Act):
    table.act(body.action)


@app.post("/chat")
def chat(body: Chat):
    table.post_chat(body.text)


@app.post("/new")
def new():
    table.next_hand()


@app.post("/tick")
def tick():
    table.tick()


def main():
    global table
    parser = argparse.ArgumentParser(description="a poker table in the browser; any mix of players")
    parser.add_argument("--p0", default="human", help="seat 0: human, a fish style, or api:<model>@<url>")
    parser.add_argument("--p1", default="station", help="seat 1: same choices")
    parser.add_argument("--key0", default="", help="api key for seat 0 (api seats only)")
    parser.add_argument("--key1", default="", help="api key for seat 1")
    parser.add_argument("--no-talk", action="store_true")
    parser.add_argument("--no-open", action="store_true", help="don't auto-open the browser")
    parser.add_argument("--port", type=int, default=8642)
    args = parser.parse_args()

    seat0, name0 = make_seat(args.p0, args.key0)
    seat1, name1 = make_seat(args.p1, args.key1)
    if seat0 is None and seat1 is None:
        raise SystemExit("two human seats need two browsers and a notion of identity — not built; keep one human")

    table = Table([seat0, seat1], [name0, name1], talk=not args.no_talk)
    url = f"http://localhost:{args.port}"
    print(f"table open at {url}  (ctrl-c to quit)")
    if not args.no_open:
        webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
