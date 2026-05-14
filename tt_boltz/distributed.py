"""Small controller/worker runtime for multi-host tt-boltz runs.

The distributed control plane intentionally moves only metadata and progress.
Inputs, model cache, MSA cache, and results live on shared storage visible at the
same paths on each host. This keeps the production path simple and predictable.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


LEASE_SECONDS = 30 * 60


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))


class ControllerStore:
    """SQLite-backed controller state."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    out_dir TEXT NOT NULL,
                    result_dir TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    run_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    worker_id TEXT,
                    lease_until REAL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT,
                    error TEXT,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (run_id, job_id)
                );
                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY,
                    host TEXT NOT NULL,
                    accelerator TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    last_seen REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    worker_id TEXT,
                    event_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )

    def create_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = payload.get("run_id") or uuid.uuid4().hex[:12]
        now = time.time()
        jobs = payload["jobs"]
        config = payload["config"]
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                    (run_id, data, out_dir, result_dir, status, config_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    payload["data"],
                    payload["out_dir"],
                    payload["result_dir"],
                    "running",
                    json.dumps(config),
                    now,
                    now,
                ),
            )
            for job in jobs:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO jobs
                        (run_id, job_id, path, status, updated_at)
                    VALUES (?, ?, ?, 'pending', ?)
                    """,
                    (run_id, job["id"], job["path"], now),
                )
            self.add_event(conn, run_id, None, {"event": "run", "run_id": run_id, "total": len(jobs)})
        return {"run_id": run_id, "total": len(jobs)}

    def add_event(self, conn: sqlite3.Connection, run_id: str, worker_id: str | None, event: dict[str, Any]) -> None:
        conn.execute(
            "INSERT INTO events (run_id, worker_id, event_json, created_at) VALUES (?, ?, ?, ?)",
            (run_id, worker_id, json.dumps(event), time.time()),
        )

    def register_worker(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO workers (worker_id, host, accelerator, device_id, label, last_seen)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    host=excluded.host,
                    accelerator=excluded.accelerator,
                    device_id=excluded.device_id,
                    label=excluded.label,
                    last_seen=excluded.last_seen
                """,
                (
                    payload["worker_id"],
                    payload["host"],
                    payload["accelerator"],
                    str(payload["device_id"]),
                    payload["label"],
                    now,
                ),
            )
        return {"ok": True}

    def lease(self, payload: dict[str, Any]) -> dict[str, Any]:
        worker = payload["worker"]
        worker_id = worker["worker_id"]
        batch_size = max(1, int(payload.get("batch_size") or 1))
        now = time.time()
        lease_until = now + LEASE_SECONDS
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO workers (worker_id, host, accelerator, device_id, label, last_seen)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    host=excluded.host,
                    accelerator=excluded.accelerator,
                    device_id=excluded.device_id,
                    label=excluded.label,
                    last_seen=excluded.last_seen
                """,
                (
                    worker_id,
                    worker["host"],
                    worker["accelerator"],
                    str(worker["device_id"]),
                    worker["label"],
                    now,
                ),
            )
            rows = conn.execute(
                """
                SELECT j.run_id, j.job_id, j.path, r.config_json
                FROM jobs j
                JOIN runs r ON r.run_id = j.run_id
                WHERE r.status = 'running'
                  AND (
                    j.status = 'pending'
                    OR (j.status = 'running' AND j.lease_until < ?)
                  )
                ORDER BY j.updated_at, j.job_id
                LIMIT ?
                """,
                (now, batch_size),
            ).fetchall()
            if not rows:
                return {"jobs": []}
            jobs = []
            run_id = rows[0]["run_id"]
            config_json = rows[0]["config_json"]
            for row in rows:
                if row["run_id"] != run_id:
                    continue
                conn.execute(
                    """
                    UPDATE jobs
                    SET status='running', worker_id=?, lease_until=?, attempts=attempts+1, updated_at=?
                    WHERE run_id=? AND job_id=?
                    """,
                    (worker_id, lease_until, now, row["run_id"], row["job_id"]),
                )
                jobs.append({"id": row["job_id"], "path": row["path"]})
            return {"run_id": run_id, "config": json.loads(config_json), "jobs": jobs}

    def record_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = payload["run_id"]
        worker_id = payload.get("worker_id")
        event = payload["event"]
        with self._lock, self._connect() as conn:
            self.add_event(conn, run_id, worker_id, event)
        return {"ok": True}

    def complete_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = payload["run_id"]
        worker_id = payload["worker_id"]
        row = payload["result"]
        status = row.get("status", "failed")
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status=?, result_json=?, error=?, lease_until=NULL, updated_at=?
                WHERE run_id=? AND job_id=?
                """,
                (status, json.dumps(row), row.get("error"), now, run_id, row["id"]),
            )
            event = payload.get("event")
            if event:
                self.add_event(conn, run_id, worker_id, event)
            pending = conn.execute(
                "SELECT COUNT(*) AS n FROM jobs WHERE run_id=? AND status IN ('pending', 'running')",
                (run_id,),
            ).fetchone()["n"]
            if pending == 0:
                failed = conn.execute(
                    "SELECT COUNT(*) AS n FROM jobs WHERE run_id=? AND status != 'ok'",
                    (run_id,),
                ).fetchone()["n"]
                run_status = "failed" if failed else "ok"
                conn.execute(
                    "UPDATE runs SET status=?, updated_at=? WHERE run_id=?",
                    (run_status, now, run_id),
                )
                self.add_event(conn, run_id, None, {"event": "run_done", "status": run_status, "failed": failed})
        return {"ok": True}

    def events(self, run_id: str, after: int) -> dict[str, Any]:
        with self._connect() as conn:
            events = [
                {"seq": row["seq"], **json.loads(row["event_json"])}
                for row in conn.execute(
                    "SELECT seq, event_json FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT 500",
                    (run_id, after),
                )
            ]
            run = conn.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()
            totals = conn.execute(
                """
                SELECT
                  COUNT(*) AS total,
                  SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END) AS ok,
                  SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed
                FROM jobs WHERE run_id=?
                """,
                (run_id,),
            ).fetchone()
        return {
            "events": events,
            "status": run["status"] if run else "missing",
            "total": totals["total"] or 0,
            "ok": totals["ok"] or 0,
            "failed": totals["failed"] or 0,
        }


class ControllerServer:
    def __init__(self, host: str, port: int, db_path: Path):
        self.store = ControllerStore(db_path)
        store = self.store

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # noqa: ANN001
                return

            def do_POST(self):  # noqa: N802
                try:
                    payload = _read_json(self)
                    if self.path == "/runs":
                        _json_response(self, 200, store.create_run(payload))
                    elif self.path == "/workers/register":
                        _json_response(self, 200, store.register_worker(payload))
                    elif self.path == "/lease":
                        _json_response(self, 200, store.lease(payload))
                    elif self.path == "/events":
                        _json_response(self, 200, store.record_event(payload))
                    elif self.path == "/complete":
                        _json_response(self, 200, store.complete_job(payload))
                    else:
                        _json_response(self, 404, {"error": "not found"})
                except Exception as exc:
                    _json_response(self, 500, {"error": str(exc)})

            def do_GET(self):  # noqa: N802
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path.startswith("/runs/") and parsed.path.endswith("/events"):
                    run_id = parsed.path.split("/")[2]
                    query = urllib.parse.parse_qs(parsed.query)
                    after = int((query.get("after") or ["0"])[0])
                    _json_response(self, 200, store.events(run_id, after))
                else:
                    _json_response(self, 404, {"error": "not found"})

        self.httpd = ThreadingHTTPServer((host, port), Handler)

    def serve_forever(self) -> None:
        self.httpd.serve_forever()


class ControllerClient:
    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"controller error {exc.code}: {body}") from exc

    def create_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/runs", payload)

    def register_worker(self, worker: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/workers/register", worker)

    def lease(self, worker: dict[str, Any], batch_size: int) -> dict[str, Any]:
        return self._request("POST", "/lease", {"worker": worker, "batch_size": batch_size})

    def event(self, run_id: str, worker_id: str, event: dict[str, Any]) -> None:
        self._request("POST", "/events", {"run_id": run_id, "worker_id": worker_id, "event": event})

    def complete(self, run_id: str, worker_id: str, row: dict[str, Any], event: dict[str, Any]) -> None:
        self._request("POST", "/complete", {"run_id": run_id, "worker_id": worker_id, "result": row, "event": event})

    def events(self, run_id: str, after: int) -> dict[str, Any]:
        return self._request("GET", f"/runs/{run_id}/events?after={after}")


class HttpProgressQueue:
    """Queue-like adapter that forwards worker progress to the controller."""

    def __init__(self, client: ControllerClient, run_id: str, worker_id: str):
        self.client = client
        self.run_id = run_id
        self.worker_id = worker_id

    def put_nowait(self, event: dict[str, Any]) -> None:
        try:
            self.client.event(self.run_id, self.worker_id, event)
        except Exception:
            pass


def job_payloads(jobs: list[Any]) -> list[dict[str, str]]:
    return [{"id": job.id, "path": str(job.path)} for job in jobs]


def worker_payload(worker: Any) -> dict[str, Any]:
    return asdict(worker) | {"label": worker.label}
