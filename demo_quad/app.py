"""Flask app + CLI for the quad-card live demo.

Run directly::

    python -m demo_quad.app --autoplay
    python demo_quad/app.py --autoplay

Or, after ``pip install -e .``::

    tt-boltz-demo-quad --autoplay

All the orchestration lives in :mod:`demo_quad.pool`; this file is just an
HTTP/CLI shell around it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
import zlib
from pathlib import Path
from typing import Iterable

# Make ``python demo_quad/app.py`` work the same as ``python -m demo_quad.app``.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, Response, jsonify, render_template, request

from demo_quad.complexes import ROTATION
from demo_quad.pool import CardPool


SSE_HEARTBEAT_SECONDS = 15.0


# ── HTTP layer ──────────────────────────────────────────────────────────────

def make_app(pool: CardPool) -> Flask:
    # Reuse the single-card demo's static assets (logo.png, etc.) so branding
    # stays consistent and we don't duplicate the binary into version control.
    shared_static = Path(__file__).resolve().parent.parent / "demo" / "static"
    app = Flask(__name__, static_folder=str(shared_static))
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            cards=[
                {
                    "card": c.snapshot.card_id,
                    "device": c.snapshot.device_id,
                    "label": c.snapshot.label,
                }
                for c in pool.cards.values()
            ],
        )

    @app.route("/status")
    def status():
        return jsonify(pool.status())

    @app.route("/start", methods=["POST"])
    def start():
        pool.play_workers()
        return jsonify({"playing": True})

    @app.route("/stop", methods=["POST"])
    def stop():
        pool.pause_workers()
        return jsonify({"playing": False})

    @app.route("/events")
    def events():
        accepts_gzip = "gzip" in request.headers.get("Accept-Encoding", "").lower()
        q = pool.subscribe()
        # Seed the connection with a complete snapshot so a fresh tab or a
        # reconnect after a network blip is immediately consistent.
        seed = {"event": "snapshot", **pool.status()}

        def stream() -> Iterable[bytes | str]:
            yield _sse(seed)
            last_beat = time.time()
            try:
                while True:
                    try:
                        ev = q.get(timeout=1.0)
                        yield _sse(ev)
                    except Exception:
                        # queue.Empty — fall through to the heartbeat check.
                        pass
                    if time.time() - last_beat > SSE_HEARTBEAT_SECONDS:
                        # SSE comment line; cheap keepalive that reverse
                        # proxies (nginx, traefik) recognize and won't buffer.
                        yield ": ping\n\n"
                        last_beat = time.time()
            except GeneratorExit:
                pass
            finally:
                pool.unsubscribe(q)

        body: Iterable[bytes | str] = stream()
        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Vary": "Accept-Encoding",
        }
        if accepts_gzip:
            body = _gzip(body)
            headers["Content-Encoding"] = "gzip"
        return Response(body, mimetype="text/event-stream", headers=headers)

    return app


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n"


def _gzip(chunks: Iterable[bytes | str]) -> Iterable[bytes]:
    """Stream-gzip an SSE body. Flushes every chunk so events arrive live."""
    compressor = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
    for chunk in chunks:
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        if not chunk:
            continue
        out = compressor.compress(chunk) + compressor.flush(zlib.Z_SYNC_FLUSH)
        if out:
            yield out
    tail = compressor.flush(zlib.Z_FINISH)
    if tail:
        yield tail


# ── CLI ─────────────────────────────────────────────────────────────────────

def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TT-Boltz quad-card live demo")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--no-fast", action="store_true",
                        help="Disable block-fp8 fast mode (slower, more accurate).")
    parser.add_argument("--sampling-steps", type=int, default=200,
                        help="Diffusion steps per prediction.")
    parser.add_argument("--linger-seconds", type=float, default=2.0,
                        help="How long to show each finished structure before the next prediction.")
    parser.add_argument("--autoplay", action="store_true",
                        help="Begin predicting immediately without waiting for /start.")
    parser.add_argument("--log-dir", default=None,
                        help="Where workers write their rotating log files. Defaults to ~/.boltz/logs.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    pool = CardPool(
        complexes=ROTATION,
        fast=not args.no_fast,
        sampling_steps=args.sampling_steps,
        linger_seconds=args.linger_seconds,
        log_dir=args.log_dir,
    )

    # Make sure we always reap workers on exit — covers SIGINT/SIGTERM,
    # interpreter shutdown, and uncaught exceptions in Flask's threads.
    import atexit
    atexit.register(pool.stop)

    def _bye(signum, _frame):
        logging.info("received signal %d, shutting down", signum)
        pool.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _bye)
    signal.signal(signal.SIGTERM, _bye)

    pool.start()
    if args.autoplay:
        pool.play_workers()

    app = make_app(pool)
    try:
        # threaded=True so SSE streams don't serialize each other and slow
        # /status etc. behind a long-running diffusion update.
        app.run(host=args.host, port=args.port, debug=False, threaded=True,
                use_reloader=False)
    finally:
        pool.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
