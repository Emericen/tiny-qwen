"""
A poker table in the browser. Standard library only; the dealer does the rules.

Play against a scripted fish or any OpenAI-compatible model:

    python -m rl.table --vs station
    python -m rl.table --vs api --model grok-4.5 --url https://api.x.ai/v1 --api-key $XAI_API_KEY

Watch two AIs play each other (both hands face up, auto-advancing):

    python -m rl.table --watch --vs api --model ... --vs2 station

Then open http://localhost:8642. The page shows the table, the action history,
each seat's running total in big blinds, and buttons for whatever the dealer
says is legal. Table talk goes in the text box next to the buttons.
"""

import argparse
import json
import threading
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rl.dealer import BB, Dealer, FishSeat, PolicySeat, STREETS, cards_str

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


def make_seat(kind, args, which=""):
    if kind == "api":
        return api_seat(getattr(args, "model" + which), getattr(args, "url" + which), getattr(args, "api_key" + which))
    return FishSeat(kind, seed=len(which))


class Table:
    """One long session of hands. Seat 0 = human (play mode) or AI A (watch mode)."""

    def __init__(self, seats, names, watch, talk):
        self.seats = seats  # [None, opponent] in play mode; [ai_a, ai_b] in watch mode
        self.names = names
        self.watch = watch
        self.talk = talk
        self.hand_no = 0
        self.totals = [0.0, 0.0]
        self.lock = threading.Lock()
        self._deal()

    def _deal(self):
        self.hand_no += 1
        self.dealer = Dealer(seed=self.hand_no, button=self.hand_no % 2, talk=self.talk)
        self._advance_ai()

    def _advance_ai(self, limit=50):
        """Let AI seats act until it's the human's turn, the hand ends, or (watch
        mode) one AI decision was made — so the page can animate step by step."""
        for _ in range(limit):
            if self.dealer.done:
                self.totals[0] += self.dealer.result()[0]
                self.totals[1] += self.dealer.result()[1]
                return
            seat = self.dealer.to_act
            if self.seats[seat] is None:
                return  # human's turn
            action, say = self.seats[seat].act(self.dealer.observation(seat), self.dealer.legal_actions(seat))
            self.dealer.step(action, say)
            if self.watch:
                return  # one decision per tick, so the viewer can follow

    def act(self, action, say):
        with self.lock:
            if self.dealer.done or self.seats[self.dealer.to_act] is not None:
                return
            try:
                self.dealer.step(action, say if self.talk else "")
            except ValueError:
                return  # stale or illegal request; the page will re-fetch the legal menu
            self._advance_ai()

    def tick(self):
        with self.lock:
            if not self.dealer.done:
                self._advance_ai()

    def next_hand(self):
        with self.lock:
            if self.dealer.done:
                self._deal()

    def state(self):
        d = self.dealer
        human_seat = 0
        show = [True, self.watch] if not d.done else [True, True]
        if self.watch:
            show = [True, True]
        history = []
        last_street = None
        for street, who, action, say in d.history:
            if street != last_street:
                history.append({"street": STREETS[street]})
                last_street = street
            history.append({"who": self.names[who], "action": action, "say": say})
        state = {
            "hand_no": self.hand_no,
            "watch": self.watch,
            "names": self.names,
            "street": STREETS[d.street],
            "board": cards_str(d.board).split() if d.board else [],
            "pot": d.pot_total(),
            "stacks": d.stack,
            "bets": d.bet,
            "hole": [cards_str(d.hole[i]).split() if show[i] else ["??", "??"] for i in range(2)],
            "button": d.button,
            "to_act": d.to_act,
            "your_turn": (not self.watch) and (not d.done) and d.to_act == human_seat,
            "legal": d.legal_actions() if not d.done else [],
            "done": d.done,
            "result": d.result() if d.done else None,
            "totals_bb": [round(t / BB, 1) for t in self.totals],
            "history": history,
            "talk": self.talk,
        }
        if d.done:
            state["result_text"] = d._result_text(0 if not self.watch else 0)
        return state


PAGE = """<!doctype html><meta charset="utf-8"><title>tiny-qwen poker</title>
<style>
  body { font-family: ui-monospace, Menlo, monospace; background:#0b3d20; color:#eee;
         max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
  h2 { font-weight: normal; }
  .card { display:inline-block; background:#fff; color:#111; border-radius:6px;
          padding:.35rem .5rem; margin:.1rem; font-size:1.3rem; min-width:1.6rem; text-align:center; }
  .red { color:#c0392b; }
  .row { margin:.6rem 0; }
  .seat { padding:.4rem .6rem; border-radius:8px; }
  .acting { background:#14532d; outline:1px solid #4ade80; }
  button { font:inherit; padding:.5rem .9rem; margin:.15rem; border-radius:8px; border:0;
           background:#eab308; cursor:pointer; }
  button:hover { background:#facc15; }
  #say { font:inherit; padding:.45rem; border-radius:8px; border:0; width:14rem; }
  #log { background:#08210f; border-radius:8px; padding:.6rem .8rem; height:11rem;
         overflow-y:auto; font-size:.85rem; }
  #log .street { color:#4ade80; margin-top:.3rem; }
  #log .say { color:#fbbf24; }
  .muted { color:#9db8a5; }
</style>
<h2>tiny-qwen poker <span id="handno" class="muted"></span></h2>
<div class="row seat" id="opp"></div>
<div class="row" id="board"></div>
<div class="row muted" id="pot"></div>
<div class="row seat" id="me"></div>
<div class="row" id="controls"></div>
<div class="row" id="log"></div>
<div class="row muted" id="score"></div>
<script>
const suits = {s:"\\u2660", h:"\\u2665", d:"\\u2666", c:"\\u2663"};
function card(c) {
  if (c === "??") return '<span class="card muted">?</span>';
  const red = "hd".includes(c[1]) ? " red" : "";
  return `<span class="card${red}">${c[0]}${suits[c[1]]}</span>`;
}
function cards(list) { return list.map(card).join(""); }
let lastLen = 0;
async function refresh() {
  const s = await (await fetch("state")).json();
  document.getElementById("handno").textContent = `hand ${s.hand_no} · ${s.street}`;
  const btn = i => (s.button === i ? " · button" : "");
  const act = i => (!s.done && s.to_act === i ? " acting" : "");
  document.getElementById("opp").className = "row seat" + act(1);
  document.getElementById("opp").innerHTML =
    `${s.names[1]}${btn(1)} · stack ${s.stacks[1]}<br>${cards(s.hole[1])}`;
  document.getElementById("me").className = "row seat" + act(0);
  document.getElementById("me").innerHTML =
    `${s.names[0]}${btn(0)} · stack ${s.stacks[0]}<br>${cards(s.hole[0])}`;
  document.getElementById("board").innerHTML = s.board.length ? cards(s.board) : '<span class="muted">— board —</span>';
  document.getElementById("pot").textContent = `pot ${s.pot} (bets ${s.bets[0]} / ${s.bets[1]})`;
  document.getElementById("score").textContent =
    `totals: ${s.names[0]} ${s.totals_bb[0]} bb · ${s.names[1]} ${s.totals_bb[1]} bb`;
  const log = document.getElementById("log");
  if (s.history.length < lastLen) log.innerHTML = "";
  for (const h of s.history.slice(lastLen === s.history.length ? lastLen : 0)) {}
  log.innerHTML = s.history.map(h =>
    h.street ? `<div class="street">— ${h.street} —</div>`
             : `<div>${h.who}: ${h.action}${h.say ? ` <span class="say">\\u201c${h.say}\\u201d</span>` : ""}</div>`
  ).join("");
  if (s.done && s.result_text) log.innerHTML += `<div class="street">${s.result_text}</div>`;
  log.scrollTop = log.scrollHeight;
  lastLen = s.history.length;
  const controls = document.getElementById("controls");
  if (s.done) {
    controls.innerHTML = '<button onclick="post(\\'new\\')">next hand</button>';
  } else if (s.your_turn) {
    controls.innerHTML = s.legal.map(a => `<button onclick="play('${a}')">${a}</button>`).join("")
      + (s.talk ? ' <input id="say" placeholder="table talk (optional)">' : "");
  } else {
    controls.innerHTML = '<span class="muted">waiting\\u2026</span>';
    if (s.watch || !s.your_turn) setTimeout(() => post("tick"), 900);
  }
}
async function play(action) {
  const say = document.getElementById("say")?.value ?? "";
  await fetch("act", {method:"POST", body: JSON.stringify({action, say})});
  refresh();
}
async function post(path) { await fetch(path, {method:"POST"}); refresh(); }
setInterval(refresh, 1200);
refresh();
</script>"""


class Handler(BaseHTTPRequestHandler):
    table = None

    def log_message(self, *args):
        pass

    def _send(self, body, kind="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/state":
            self._send(json.dumps(self.table.state()))
        else:
            self._send(PAGE, "text/html")

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}") if length else {}
        if self.path == "/act":
            self.table.act(body.get("action", ""), body.get("say", ""))
        elif self.path == "/new":
            self.table.next_hand()
        elif self.path == "/tick":
            self.table.tick()
        self._send("{}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vs", default="station", help="opponent: station | nit | maniac | random | api")
    parser.add_argument("--model", default="")
    parser.add_argument("--url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--watch", action="store_true", help="two AIs play each other, hands face up")
    parser.add_argument("--vs2", default="station", help="watch mode: seat 0's driver (same choices as --vs)")
    parser.add_argument("--model2", default="")
    parser.add_argument("--url2", default="")
    parser.add_argument("--api-key2", default="")
    parser.add_argument("--no-talk", action="store_true")
    parser.add_argument("--port", type=int, default=8642)
    args = parser.parse_args()

    opponent = make_seat(args.vs, args)
    opp_name = args.model if args.vs == "api" else f"fish ({args.vs})"
    if args.watch:
        seat0 = make_seat(args.vs2, args, which="2")
        name0 = args.model2 if args.vs2 == "api" else f"fish ({args.vs2})"
        table = Table([seat0, opponent], [name0, opp_name], watch=True, talk=not args.no_talk)
    else:
        table = Table([None, opponent], ["you", opp_name], watch=False, talk=not args.no_talk)

    Handler.table = table
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"table open at {url}  (ctrl-c to quit)")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
