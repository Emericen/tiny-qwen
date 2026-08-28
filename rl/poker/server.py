"""
The server: a small FastAPI wrapper whose state is a map of Games.

    python -m rl.poker.server                         # you vs a calling-station fish
    python -m rl.poker.server --p0 maniac --p1 nit    # spectate two fish

The game does not know who sits behind a seat — every action arrives as
POST /act {seat, total} (total = your cumulative chips this hand, null = fold).
The browser is one client; the fish NPCs at the bottom of this file are
another; model players live with the training code (rl/model_player.py).
Views stream OUT over SSE, one snapshot per game event, filtered per seat.

The CLI builds table "main"; POST /create adds more (any player count — the
browser page renders 2-player tables; others are for API consumers like the
eval harness).
"""

import argparse
import asyncio
import contextlib
import json
import random
import threading
import webbrowser
import zlib
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from rl.poker.game import Game

PAGE = Path(__file__).with_name("table.html")

tables: dict[str, Game] = {}
clocks: list[asyncio.Task] = []  # every running task: game clocks and fish NPCs
pending_fish: list[tuple[str, int, str, float]] = []  # (table, seat, style, pace) to start with the app


class Act(BaseModel):
    seat: int
    total: int | None  # required but nullable: null = fold
    table: str = "main"


class Chat(BaseModel):
    seat: int
    text: str
    table: str = "main"


class NewHand(BaseModel):
    table: str = "main"


class Create(BaseModel):
    names: list[str]
    chips: int = 200
    seed: int = 0
    talk: bool = True
    pause: float = 0.0  # seconds a finished hand stays up (created tables auto-deal)
    fish: dict[int, str] = {}  # optional NPCs: seat index -> style
    pace: float = 0.0  # seconds a fish thinks before acting


def get_table(name: str) -> Game:
    if name not in tables:
        raise HTTPException(404, f"no table {name!r}")
    return tables[name]


def start_clock(name: str):
    clocks.append(asyncio.create_task(tables[name].run()))


def start_fish(table: str, seat: int, style: str, pace: float):
    clocks.append(asyncio.create_task(fish(tables[table], seat, style, pace)))


def build_app() -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app):
        for name in tables:
            start_clock(name)
        for table, seat, style, pace in pending_fish:
            start_fish(table, seat, style, pace)
        yield
        for task in clocks:
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*clocks)

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    def page():
        return FileResponse(PAGE)  # read per request, so editing table.html only needs a refresh

    @app.get("/tables")
    def list_tables():
        return {name: {"names": g.names, "hand_no": g.hand_no, "totals_bb": g.totals} for name, g in tables.items()}

    @app.post("/create")
    async def create(body: Create):
        if len(body.names) < 2:
            raise HTTPException(400, "a table needs at least 2 seats")
        for seat, style in body.fish.items():
            if not 0 <= seat < len(body.names):
                raise HTTPException(400, f"fish seat {seat} out of range")
            if style not in FISH_STYLES:
                raise HTTPException(400, f"unknown fish style {style!r}; use one of {FISH_STYLES}")
        name = f"t{len(tables)}"
        tables[name] = Game(body.names, chips=body.chips, seed=body.seed, talk=body.talk,
                            auto_deal=True, pause=body.pause)
        start_clock(name)
        for seat, style in body.fish.items():
            start_fish(name, seat, style, body.pace)
        return {"table": name}

    @app.get("/events")
    async def events(seat: int | None = None, table: str = "main"):
        """The stream. seat = play that seat's view; no seat = spectator.
        Each message is a full state snapshot for that role."""
        game = get_table(table)

        async def stream():
            version = 0
            yield f"data: {json.dumps(game.state(seat))}\n\n"
            while True:
                version = await game.wait_past(version)
                yield f"data: {json.dumps(game.state(seat))}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/act")
    async def act(body: Act):
        get_table(body.table).give(body.seat, body.total)
        return {"ok": True}

    @app.post("/chat")
    async def chat(body: Chat):
        await get_table(body.table).post_chat(body.seat, body.text)
        return {"ok": True}

    @app.post("/new")
    async def new(body: NewHand):
        get_table(body.table).request_new()
        return {"ok": True}

    return app


def main():
    parser = argparse.ArgumentParser(description="a poker server; humans and fish at table 'main'")
    parser.add_argument("--p0", default="human", help="seat 0: human or a fish style "
                        f"({', '.join(FISH_STYLES)}); model players connect via rl.model_player")
    parser.add_argument("--p1", default="station", help="seat 1: same choices")
    parser.add_argument("--chips", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pace", type=float, default=0.8, help="seconds a fish thinks before acting")
    parser.add_argument("--no-talk", action="store_true")
    parser.add_argument("--no-open", action="store_true", help="don't auto-open the browser")
    parser.add_argument("--port", type=int, default=8642)
    args = parser.parse_args()

    names = []
    humans = []
    for seat, spec in enumerate([args.p0, args.p1]):
        if spec == "human":
            humans.append(seat)
            names.append("you")
        elif spec in FISH_STYLES:
            names.append(f"fish ({spec})")
            pending_fish.append(("main", seat, spec, args.pace))
        else:
            raise SystemExit(f"unknown seat spec {spec!r}; use human or one of {FISH_STYLES}")

    tables["main"] = Game(names, chips=args.chips, seed=args.seed, talk=not args.no_talk,
                          auto_deal=not humans, pause=args.pace * 3)
    base = f"http://localhost:{args.port}/"
    for seat in humans:
        print(f"seat {seat}: {base}?seat={seat}")
    url = f"{base}?seat={humans[0]}" if humans else base
    print(f"table open at {url}  (ctrl-c to quit)")
    if not args.no_open:
        threading.Timer(0.8, webbrowser.open, (url,)).start()
    uvicorn.run(build_app(), host="127.0.0.1", port=args.port, log_level="warning")


# ---------------------------------------------------------------------------
# fish: hardcoded NPCs — the one convenience the server keeps in-house
# ---------------------------------------------------------------------------

FISH_STYLES = ("station", "nit", "maniac", "random")


def fish_pick(state: dict, style: str, rng: random.Random) -> int | None:
    """One decision from the menu — the same food the browser buttons eat."""
    entries = {entry["action"].split()[0]: entry for entry in state["legal"]}
    raises = [e["total"] for e in state["legal"] if e["action"].split()[0] in ("raise", "allin")]
    stay = (entries.get("check") or entries["call"])["total"]
    if style == "station":  # calls everything, never raises
        return stay
    if style == "nit":  # folds to any bet, checks otherwise
        return None if "fold" in entries else stay
    if style == "maniac":  # escalates whenever it can
        return rng.choice(raises) if raises else stay
    r = rng.random()  # "random"
    if r < 0.1 and "fold" in entries:
        return None
    if raises and r < 0.5:
        return rng.choice(raises)
    return stay


async def fish(game: Game, seat: int, style: str, pace: float = 0.0):
    """An NPC at `seat`: wait for the turn, pick by style, deposit the number.
    In-process, but it eats only the public surface (state / give / wait_past)."""
    rng = random.Random(zlib.crc32(f"{style}:{seat}".encode()))
    version = 0
    while True:
        state = game.state(seat)
        if state["your_turn"]:
            if pace:
                await asyncio.sleep(pace)
            game.give(seat, fish_pick(state, style, rng))
        version = await game.wait_past(version)


if __name__ == "__main__":
    main()
