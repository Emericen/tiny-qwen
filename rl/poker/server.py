"""
The door. Launches the game and streams it.

    python -m rl.poker.server                                  # you vs a calling station
    python -m rl.poker.server --p0 maniac --p1 nit             # spectate two fish
    python -m rl.poker.server --p1 api:grok-4.5@https://api.x.ai/v1 --key1 $XAI_API_KEY

The game is a data object built here and run as one background task; the
browser is a subscriber. Views stream OUT over SSE (GET /events — one full
state snapshot per game event, filtered per connection); actions come IN as
plain POSTs. Any number of tabs can watch; none of them drive the clock.
"""

import argparse
import asyncio
import contextlib
import json
import threading
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from rl.poker.game import Table
from rl.poker.seat import Seat

PAGE = Path(__file__).with_name("table.html")


class Act(BaseModel):
    action: str


class Chat(BaseModel):
    text: str


def build_app(table: Table) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app):
        clock = asyncio.create_task(table.run())
        yield
        clock.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await clock

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    def page():
        return FileResponse(PAGE)  # read per request, so editing table.html only needs a refresh

    @app.get("/events")
    async def events(seat: int | None = None):
        """The stream. No seat param: you are the human if one exists, else a
        spectator. Each message is a full state snapshot for that role."""
        role = seat if seat is not None else table.human
        queue = table.subscribe()

        async def stream():
            try:
                yield f"data: {json.dumps(table.state(role))}\n\n"
                while True:
                    await queue.get()
                    yield f"data: {json.dumps(table.state(role))}\n\n"
            finally:
                table.unsubscribe(queue)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/act")
    def act(body: Act):
        table.give(body.action)
        return {"ok": True}

    @app.post("/chat")
    def chat(body: Chat):
        table.post_chat(body.text)
        return {"ok": True}

    @app.post("/new")
    def new():
        table.request_new()
        return {"ok": True}

    return app


def main():
    parser = argparse.ArgumentParser(description="a poker table in the browser; any mix of players")
    parser.add_argument("--p0", default="human", help="seat 0: human, a fish style, or api:<model>@<url>")
    parser.add_argument("--p1", default="station", help="seat 1: same choices")
    parser.add_argument("--key0", default="", help="api key for seat 0 (api seats only)")
    parser.add_argument("--key1", default="", help="api key for seat 1")
    parser.add_argument("--chips", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pace", type=float, default=0.8, help="seconds between AI actions")
    parser.add_argument("--no-talk", action="store_true")
    parser.add_argument("--no-open", action="store_true", help="don't auto-open the browser")
    parser.add_argument("--port", type=int, default=8642)
    args = parser.parse_args()

    seats = [Seat(args.p0, args.key0), Seat(args.p1, args.key1)]
    if all(seat.is_human for seat in seats):
        raise SystemExit("two human seats need two browsers and a notion of identity — not built; keep one human")

    table = Table(seats, chips=args.chips, seed=args.seed, talk=not args.no_talk, pace=args.pace)
    url = f"http://localhost:{args.port}"
    print(f"table open at {url}  (ctrl-c to quit)")
    if not args.no_open:
        threading.Timer(0.8, webbrowser.open, (url,)).start()
    uvicorn.run(build_app(table), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
