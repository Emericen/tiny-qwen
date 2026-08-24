"""
Every seat speaks OpenAI tool calls. The model takes its action by calling the
`act` tool — in training (TRL's environment loop), in the evals, and at the
browser table — so a number measured on one path is comparable with the rest,
and "invalid" means the same thing everywhere: no act call, or an illegal action.

Qwen3.5 serializes tool calls as XML inside the chat template; transformers
parses that back into standard tool_calls. Some transformers builds accept the
new-style response template yet silently return no tool_calls, so we verify the
exact parse path once and fall back to the legacy response schema — the same
dance rl/train.py does for the trainer.
"""

import json
import re

from rl.game.dealer import default_action, parse_reply

SYSTEM = (
    "You are playing heads-up no-limit Texas hold'em for chips. "
    "Take your action by calling the `act` tool with one legal action copied exactly from the list. "
    "Never answer in plain text; every decision must be an `act` call."
)

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
                    "description": 'One legal action copied exactly from the list, for example "call", "raise 6", "bet 4" or "allin".',
                },
                "say": {"type": "string", "description": "Optional table talk your opponent will see."},
            },
            "required": ["action"],
        },
    },
}


def tool_call_args(message):
    """(action, say) out of an assistant message dict. (None, "") when it didn't call act."""
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        if function.get("name") != "act":
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None, ""
        if not isinstance(arguments, dict):
            return None, ""
        action = str(arguments.get("action") or "").strip()
        say = str(arguments.get("say") or "").strip()
        return (action or None), say
    # Some models at temperature write the call as TEXT instead of a structured
    # entry (Grok 4.6: bare {"name": "act", ...} JSON, sometimes <tool_call>-wrapped,
    # "arguments" or "parameters"). The bench measures poker, not formatting, so
    # decode those too — this path never fires for a well-formed response.
    content = (message.get("content") or "").strip()
    blob_match = re.search(r"\{.*\}", content, re.DOTALL)
    if blob_match:
        try:
            blob = json.loads(blob_match.group(0))
        except json.JSONDecodeError:
            return None, ""
        if isinstance(blob, dict) and blob.get("name") == "act":
            arguments = blob.get("arguments") or blob.get("parameters") or {}
            if isinstance(arguments, dict):
                action = str(arguments.get("action") or "").strip()
                say = str(arguments.get("say") or "").strip()
                return (action or None), say
    return None, ""


class ToolSeat:
    """Wraps `generate` (observation text -> assistant message dict) as a seat.

    The action argument is still validated against the legal menu; a missing
    call or an illegal action counts as invalid and becomes the cheapest legal
    action, exactly like the trainer's env does.
    """

    def __init__(self, generate):
        self.generate = generate
        self.invalid = 0
        self.decisions = 0

    def act(self, observation, legal):
        message = self.generate(observation)
        self.decisions += 1
        action, say = tool_call_args(message)
        chosen = None
        if action:
            chosen, _ = parse_reply(action, legal)
        if chosen is None:
            self.invalid += 1
            # An invalid reply becomes a forced default action — that biases results, so
            # every one must be visible in the log, not just a count at the end.
            shown = {k: message.get(k) for k in ("content", "tool_calls") if message.get(k)}
            print(f"[invalid reply #{self.invalid}] {json.dumps(shown, default=str)[:400]}", flush=True)
            chosen = default_action(legal)
        return chosen, say


def configure_tool_parsing(tokenizer):
    """Make parse_message work for Qwen3.5's XML tool calls; see module docstring."""
    from trl.chat_template_utils import add_response_schema, parse_response, qwen3_5_schema

    try:
        add_response_schema(tokenizer)
    except ValueError:
        # TRL only recognizes templates it knows. Qwen3.8 emits byte-identical tool-call XML
        # to Qwen3.5, so borrow the parsing config from a tokenizer TRL does recognize —
        # this tracks whichever attribute the installed transformers wants.
        from transformers import AutoTokenizer

        donor = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
        add_response_schema(donor)
        tokenizer.response_template = getattr(donor, "response_template", None)
        tokenizer.response_schema = getattr(donor, "response_schema", None)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "call the act tool"}], tools=[ACT_TOOL], add_generation_prompt=True, tokenize=True
    )
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids["input_ids"]
    sample = "<think>\n\n</think>\n\n<tool_call>\n<function=act>\n<parameter=action>\ncall\n</parameter>\n</function>\n</tool_call>"
    sample_ids = tokenizer(sample + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

    def parses():
        try:
            return bool(parse_response(tokenizer, sample_ids, prefix=prompt_ids).get("tool_calls"))
        except Exception:
            return False

    if not parses():
        tokenizer.response_template = None
        tokenizer.response_schema = qwen3_5_schema
    if not parses():
        raise RuntimeError("tokenizer cannot parse Qwen3.5 tool calls")


def parse_message(tokenizer, completion_ids, prompt_ids):
    """Completion token ids -> assistant message dict, via the configured parser."""
    from trl.chat_template_utils import parse_response

    return parse_response(tokenizer, completion_ids, prefix=prompt_ids)
