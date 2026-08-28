"""
The model player: an LLM policy as a CLIENT of the poker server. This is
training-side code — the game neither imports nor knows it; it eats the same
SSE snapshots the browser eats and POSTs the same {seat, total}.

    python -m rl.model_player grok-4.5@https://api.x.ai/v1 --seat 1 --key $XAI_API_KEY

Tool-call native: ACT_TOOL rides every request with tool_choice forced; a bad
call gets ONE repair re-ask, then the cheapest legal default, billed on
self.invalid — the counter the eval harness reads. Table talk rides the
'say' argument and is POSTed to /chat.
"""

import argparse
import asyncio
import json

import httpx

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


class ModelPlayer:
    """decide(state) -> (total, say) via one forced tool call against any
    OpenAI-compatible endpoint; play() is the loop that seats it at a table."""

    def __init__(self, model: str, api: str, key: str = "", temperature: float = 1.0, max_tokens: int = 300):
        self.model, self.api, self.key = model, api, key
        self.temperature, self.max_tokens = temperature, max_tokens
        self.name = model.split("/")[-1]
        self.decisions = 0
        self.invalid = 0

    async def decide(self, state: dict) -> tuple[int | None, str]:
        self.decisions += 1
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": self.render_prompt(state)},
        ]
        for attempt in range(2):  # one honest try, one repair re-ask
            message = await self._call(messages)
            action, say = self._extract(message)
            if action is not None:
                try:
                    total = self.to_total(state, action)
                    if self._legal(state, total):
                        return total, say
                    error = f"'{action}' is not legal here; legal actions: " + ", ".join(
                        entry["action"] for entry in state["legal"]
                    )
                except ValueError:
                    error = f"could not understand action {action!r}"
            else:
                error = "no act tool call found in the reply"
            if attempt == 0:  # the repair pattern: attach the failure, ask again
                messages.append({
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": message.get("tool_calls") or [],
                })
                messages.append({
                    "role": "user",
                    "content": f"That was invalid: {error}. Call the act tool with one legal action.",
                })
        self.invalid += 1
        facing_bet = any(entry["action"] == "fold" for entry in state["legal"])
        return (None if facing_bet else state["match"]), ""  # cheapest legal thing

    async def play(self, server: str, table: str, seat: int):
        """The client loop: watch the stream; on a fresh decision point that is
        our turn, decide and POST. One connection, runs until cancelled."""
        base = server.rstrip("/")
        last_point = None
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "GET", f"{base}/events", params={"seat": seat, "table": table}
            ) as response:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    state = json.loads(line[len("data: "):])
                    if not state["your_turn"]:
                        continue
                    point = (state["hand_no"], state["street"], state["pot"] + sum(state["bets"]), state["match"])
                    if point == last_point:
                        continue  # a chat ping on a spot we already answered
                    last_point = point
                    total, say = await self.decide(state)
                    if say:
                        await client.post(f"{base}/chat", json={"seat": seat, "text": say, "table": table})
                    await client.post(f"{base}/act", json={"seat": seat, "total": total, "table": table})

    async def _call(self, messages: list[dict]) -> dict:
        """Transport, nothing else. Returns the assistant message dict."""
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
                self.api.rstrip("/") + "/chat/completions",
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
    def to_total(state: dict, action: str) -> int | None:
        """The declared action string -> the engine's number. Vocabulary
        application, not parsing — the string comes from a schema field.
        ValueError on an unknown word."""
        words = action.strip().lower().replace("-", "").split()
        if not words:
            raise ValueError(action)
        verb = words[0]
        if verb == "fold":
            return None
        if verb in ("check", "call"):
            return state["match"]
        if verb == "allin":
            return state["escalation"][1] if state["escalation"] else state["match"]
        if verb in ("raise", "bet"):
            for word in words[1:]:
                if word.isdigit():
                    return int(word)
        raise ValueError(action)

    @staticmethod
    def _legal(state: dict, total: int | None) -> bool:
        if total is None or total == state["match"]:
            return True
        escalation = state["escalation"]
        return escalation is not None and escalation[0] <= total <= escalation[1]

    @staticmethod
    def render_prompt(state: dict) -> str:
        """One SSE snapshot (ASCII cards already) -> the text a model sees."""
        n = len(state["names"])
        small_blind, big_blind = state["blinds"]
        kind = "Heads-up" if n == 2 else f"{n}-player"
        lines = [f"{kind} no-limit hold'em. Blinds {small_blind}/{big_blind}. 1 bb = {big_blind} chips."]
        lines.append(f"You are {state['names'][state['you']]} (seat {state['you']}).")
        lines.append(f"Your hole cards: {' '.join(state['hole'][state['you']])}")
        board = " ".join(state["board"]) or "(none yet)"
        lines.append(f"Street: {STREETS[state['street']]}. Board: {board}")
        pot = state["pot"] + sum(state["bets"])
        lines.append(f"Pot: {pot}. Your stack: {state['stacks'][state['you']]}.")
        cost = next((entry["n"] for entry in state["legal"] if entry["action"] == "call"), 0)
        if cost:
            lines.append(f"To call: {cost}.")
        lines.append("Legal actions: " + ", ".join(entry["action"] for entry in state["legal"]))
        if state["talk"]:
            talk = [entry for entry in state["chat"] if entry["kind"] == "talk"][-8:]
            if talk:
                lines.append("Table talk so far:")
                lines.extend(f"{entry['who']}: {entry['text']}" for entry in talk)
            lines.append("You may talk to the table via the 'say' argument.")
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="seat an LLM at a poker table")
    parser.add_argument("spec", help="<model>@<openai-compatible base url>, e.g. grok-4.5@https://api.x.ai/v1")
    parser.add_argument("--seat", type=int, required=True)
    parser.add_argument("--key", default="")
    parser.add_argument("--server", default="http://localhost:8642")
    parser.add_argument("--table", default="main")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    model, _, api = args.spec.rpartition("@")
    if not model or not api:
        raise SystemExit(f"bad spec {args.spec!r}; use <model>@<base-url>")
    player = ModelPlayer(model, api, key=args.key, temperature=args.temperature)
    asyncio.run(player.play(args.server, args.table, args.seat))


if __name__ == "__main__":
    main()
