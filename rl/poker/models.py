"""
The model adapter: renders a seat's view into a prompt, calls an
OpenAI-compatible endpoint, and decodes the reply into an action.

This is the LLM's table.html — one of two sibling renderers of the same view
dict that Table assembles (the other renders pixels). Nothing here is poker;
nothing in game.py knows models exist.

Decoding is regex-over-text as the reliability floor: structured tool calls
are the right primary path, but providers measurably leak calls into plain
content (Grok 4.6: ~40% at temperature — rl/runs-0824, incident 4), so the
text decoder is what everything can fall back on.
"""

import json
import re
import urllib.request

from rl.poker.game import BIG_BLIND, SMALL_BLIND, card_ascii

STREETS = ("preflop", "flop", "turn", "river")

SYSTEM = (
    "You are playing no-limit Texas hold'em for chips. "
    "Read the state and reply with exactly one legal action copied from the list. "
    "You may add one more line starting with 'say:' to talk to the table."
)


def render_prompt(view: dict) -> str:
    """view -> the text a model sees. ASCII cards — the model's format."""
    kind = "Heads-up" if view["player_count"] == 2 else f"{view['player_count']}-player"
    lines = [f"{kind} no-limit hold'em. Blinds {SMALL_BLIND}/{BIG_BLIND}. 1 bb = {BIG_BLIND} chips."]
    lines.append(f"You are {view['name']} (seat {view['seat']}).")
    lines.append(f"Your hole cards: {' '.join(card_ascii(c) for c in view['hole'])}")
    board = " ".join(card_ascii(c) for c in view["board"]) or "(none yet)"
    lines.append(f"Street: {STREETS[view['street']]}. Board: {board}")
    lines.append(f"Pot: {view['pot']}. Your stack: {view['stacks'][view['seat']]}.")
    if view["cost"] > 0:
        lines.append(f"To call: {view['cost']}.")
    lines.append("Legal actions: " + ", ".join(entry["action"] for entry in view["menu"]))
    lines.append("Reply with exactly one legal action from the list.")
    if view["talk"]:
        if view["chat"]:
            lines.append("Table talk so far:")
            lines.extend(f"{entry['who']}: {entry['text']}" for entry in view["chat"])
        lines.append("You may add table talk on a second line: say: <message>")
    return "\n".join(lines)


ACTION_RE = re.compile(r"\b(fold|check|call|all-?in|raise\s+\d+|bet\s+\d+)\b", re.IGNORECASE)
SAY_RE = re.compile(r"say:\s*(.+)", re.IGNORECASE)


def parse_reply(text: str, match: int, escalation, cost: int = 0) -> tuple[int | None, str, bool]:
    """Free model text -> (commit_to, say, valid). The last action mentioned
    wins; no valid action -> the cheapest legal thing (fold when matching
    costs chips, otherwise check), reported as valid=False so harnesses can
    count it. `escalation` accepts a range, an (lo, hi) pair, or None."""
    if escalation:
        lo, hi = (escalation.start, escalation[-1]) if isinstance(escalation, range) else tuple(escalation)
    else:
        lo = hi = None
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
            chosen, found = (hi if hi is not None else match), True
        elif word == "fold":
            chosen, found = None, True
        else:
            n = int(word.split()[1])
            if lo is not None and lo <= n <= hi:
                chosen, found = n, True
    if not found:
        chosen = None if cost > 0 else match
    return chosen, say, found


class ModelSeat:
    """A seat driven by any OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, model: str, url: str, api_key: str = "", temperature: float = 1.0, max_tokens: int = 200):
        self.model, self.url, self.api_key = model, url, api_key
        self.temperature, self.max_tokens = temperature, max_tokens

    def _complete(self, prompt: str) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
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
            return json.load(response)["choices"][0]["message"]["content"]

    def act(self, view: dict) -> tuple[int | None, str]:
        reply = self._complete(render_prompt(view))
        total, say, _ = parse_reply(reply, view["match"], view["escalation"], view["cost"])
        return total, say
