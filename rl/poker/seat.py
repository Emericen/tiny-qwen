"""
The players. game.py is the game; this file is everyone who sits at it.

One class, three kinds, chosen by spec string:
    "human"                      the browser: clicks arrive via give(), act()
                                 awaits the inbox until one lands
    "station|nit|maniac|random"  scripted fish — exploitable on purpose
    "api:<model>@<base-url>"     an LLM over any OpenAI-compatible endpoint

The model path is tool-call native: ACT_TOOL rides every request with
tool_choice forced, and the answer is read from message.tool_calls — never
parsed from prose. A missing or illegal call gets ONE re-ask with the error
attached (the industry-standard repair pattern), then the cheapest legal
default, counted on self.invalid so evals can bill the failure.

The Seat protocol is a single coroutine: await act(view) -> (commit_to, say).
urllib-free, thread-free: the human awaits a queue, the model awaits httpx.
game.py never imports this file; server.py wires Seats to chairs.
"""

import asyncio
import json
import random
import zlib

from rl.poker.game import BIG_BLIND, SMALL_BLIND, card_ascii

FISH_STYLES = ("station", "nit", "maniac", "random")
STREETS = ("preflop", "flop", "turn", "river")

SYSTEM = "You are playing no-limit Texas hold'em for chips."  # the schema is the instruction

ACT_TOOL = {
    "type": "function",
    "function": {
        "name": "act",
        "description": "Take your poker action.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "One legal action copied exactly from the list, e.g. 'call' or 'raise 12'.",
                },
                "say": {
                    "type": "string",
                    "description": "Optional table talk the other players will see.",
                },
            },
            "required": ["action"],
        },
    },
}


class Seat:
    """A way of answering views. Table awaits act(view); everything else is
    the private machinery of one of the three kinds."""

    def __init__(self, spec: str, key: str = "", temperature: float = 1.0, max_tokens: int = 300):
        self.spec = spec
        self.decisions = 0
        self.invalid = 0
        if spec == "human":
            self.kind, self.name = "human", "you"
            self.inbox: asyncio.Queue = asyncio.Queue()
        elif spec.startswith("api:"):
            model, _, url = spec[4:].rpartition("@")
            if not model or not url:
                raise SystemExit(f"bad api seat spec {spec!r}; use api:<model>@<base-url>")
            self.kind, self.name = "model", model.split("/")[-1]
            self.model, self.url, self.key = model, url, key
            self.temperature, self.max_tokens = temperature, max_tokens
        else:
            style = spec.removeprefix("fish:")
            if style not in FISH_STYLES:
                raise SystemExit(
                    f"unknown seat spec {spec!r}; use human, {', '.join(FISH_STYLES)}, or api:<model>@<url>"
                )
            self.kind, self.name = "fish", f"fish ({style})"
            self.style = style
            self.rng = random.Random(zlib.crc32(spec.encode()) % 1000)

    @property
    def is_human(self) -> bool:
        return self.kind == "human"

    # -- the protocol -------------------------------------------------------

    async def act(self, view: dict) -> tuple[int | None, str]:
        self.decisions += 1
        if self.kind == "human":
            return await self._act_human(view)
        if self.kind == "fish":
            return self._act_fish(view)
        return await self._act_model(view)

    def give(self, action: str):
        """Human only: the bridge deposits the browser's click here."""
        self.inbox.put_nowait(action)

    # -- kind: human --------------------------------------------------------

    async def _act_human(self, view: dict) -> tuple[int | None, str]:
        action = await self.inbox.get()
        try:
            return self.action_to_total(view, action), ""
        except ValueError:
            return view["match"], ""  # garbage click; the engine re-validates anyway

    # -- kind: fish ---------------------------------------------------------

    def _act_fish(self, view: dict) -> tuple[int | None, str]:
        match, cost, escalation = view["match"], view["cost"], view["escalation"]
        if self.style == "station":  # calls everything, never raises
            return match, ""
        if self.style == "nit":  # folds to any bet, checks otherwise
            return (None if cost > 0 else match), ""
        if self.style == "maniac":  # escalates when it can, else calls
            if escalation:
                lo, hi = escalation
                return self.rng.choice([lo, (lo + hi) // 2, hi]), ""
            return match, ""
        r = self.rng.random()  # "random"
        if r < 0.1 and cost > 0:
            return None, ""
        if escalation and r < 0.5:
            return self.rng.randint(*escalation), ""
        return match, ""

    # -- kind: model --------------------------------------------------------

    async def _act_model(self, view: dict) -> tuple[int | None, str]:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": self.render_prompt(view)},
        ]
        for attempt in range(2):  # one honest try, one repair re-ask
            message = await self._call(messages)
            action, say = self._extract(message)
            if action is not None:
                try:
                    total = self.action_to_total(view, action)
                    if self._legal(view, total):
                        return total, say
                    error = f"'{action}' is not legal here; legal actions: " + ", ".join(
                        entry["action"] for entry in view["menu"]
                    )
                except ValueError:
                    error = f"could not understand action {action!r}"
            else:
                error = "no act tool call found in the reply"
            if attempt == 0:  # the repair pattern: attach the failure, ask again
                messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": message.get("tool_calls") or []})
                messages.append({"role": "user", "content": f"That was invalid: {error}. Call the act tool with one legal action."})
        self.invalid += 1
        return (None if view["cost"] > 0 else view["match"]), ""  # cheapest legal thing

    async def _call(self, messages: list[dict]) -> dict:
        """Transport, nothing else. Returns the assistant message dict."""
        import httpx  # lazy: the trainer imports this file without transport deps

        body = {
            "model": self.model,
            "messages": messages,
            "tools": [ACT_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "act"}},
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        async with httpx.AsyncClient(timeout=240) as client:
            response = await client.post(
                self.url.rstrip("/") + "/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self.key}"},
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]

    @staticmethod
    def _extract(message: dict) -> tuple[str | None, str]:
        """message -> (action string, say). Reads tool_calls; as a repair floor,
        accepts a bare JSON object in content (providers leak calls there —
        Grok 4.6 ~40%, rl/runs-0824 incident 4). Never parses prose."""
        calls = message.get("tool_calls") or []
        if calls:
            try:
                args = json.loads(calls[0]["function"]["arguments"] or "{}")
                return args.get("action"), str(args.get("say") or "")
            except (json.JSONDecodeError, KeyError, TypeError):
                return None, ""
        content = (message.get("content") or "").strip()
        if content.startswith("{"):  # format repair, not prose parsing
            try:
                args = json.loads(content)
                args = args.get("arguments", args) if isinstance(args, dict) else {}
                if isinstance(args, str):
                    args = json.loads(args)
                return args.get("action"), str(args.get("say") or "")
            except (json.JSONDecodeError, AttributeError, TypeError):
                return None, ""
        return None, ""

    @staticmethod
    def _legal(view: dict, total: int | None) -> bool:
        if total is None or total == view["match"]:
            return True
        escalation = view["escalation"]
        return escalation is not None and escalation[0] <= total <= escalation[1]

    # -- shared vocabulary / renderer (pure; trainer and eval import these) --

    @staticmethod
    def action_to_total(view: dict, action: str) -> int | None:
        """The declared action string -> the engine's number. Vocabulary
        application, not parsing — the string comes from a schema field.
        fold -> None; check/call -> match; allin -> escalation top;
        'raise N' / 'bet N' (with optional 'to') -> N. ValueError otherwise."""
        words = action.strip().lower().replace("-", "").split()
        if not words:
            raise ValueError(action)
        verb = words[0]
        if verb == "fold":
            return None
        if verb in ("check", "call"):
            return view["match"]
        if verb == "allin":
            return view["escalation"][1] if view["escalation"] else view["match"]
        if verb in ("raise", "bet"):
            for word in words[1:]:
                if word.isdigit():
                    return int(word)
        raise ValueError(action)

    @staticmethod
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
        if view["talk"]:
            if view["chat"]:
                lines.append("Table talk so far:")
                lines.extend(f"{entry['who']}: {entry['text']}" for entry in view["chat"])
            lines.append("You may talk to the table via the 'say' argument.")
        return "\n".join(lines)
