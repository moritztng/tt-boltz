"""HTTP scheduler that dispatches Boltz-2 prediction jobs to local and remote
worker processes. Inputs ship to workers and outputs ship back over the wire,
so no shared filesystem is required."""

from __future__ import annotations

import base64
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


# Map the engine's fine-grained stage words (emitted by the worker and the
# model's progress callback) to the coarse pipeline phase a biologist follows.
_STAGE_MAP = {
    "loading": "prepare", "featuriz": "prepare", "prep": "prepare",
    "msa": "msa",
    "trunk": "fold", "pairformer": "fold", "diffusion": "fold", "sampling": "fold",
    "affinity": "score",
    "saving": "save", "writing": "save",
}


def _coarse_stage(event: dict[str, Any]) -> str | None:
    """The coarse phase an event implies, or None if it isn't a stage signal."""
    if event.get("event") == "start":
        return "prepare"
    s = str(event.get("stage") or "").lower()
    for key, phase in _STAGE_MAP.items():
        if key in s:
            return phase
    return None


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
                    owner TEXT,
                    model TEXT,
                    config_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    run_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    name TEXT NOT NULL DEFAULT '',
                    input_b64 TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    stage TEXT,
                    worker_id TEXT,
                    lease_until REAL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT,
                    outputs_json TEXT,
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
            # Migrate pre-existing tables that lack newer columns.
            for stmt in ("ALTER TABLE jobs ADD COLUMN stage TEXT",
                         "ALTER TABLE runs ADD COLUMN owner TEXT",
                         "ALTER TABLE runs ADD COLUMN model TEXT"):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass  # column already present

    def create_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = payload.get("run_id") or uuid.uuid4().hex[:12]
        now = time.time()
        jobs = payload["jobs"]
        config = payload["config"]
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                    (run_id, data, out_dir, result_dir, status, owner, model, config_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    payload["data"],
                    payload["out_dir"],
                    payload["result_dir"],
                    "running",
                    payload.get("owner"),  # opaque fairness key (hash of the user's session, never the secret)
                    config.get("model"),   # which model these jobs need — drives worker model-affinity
                    json.dumps(config),
                    now,
                    now,
                ),
            )
            for job in jobs:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO jobs
                        (run_id, job_id, name, input_b64, status, updated_at)
                    VALUES (?, ?, ?, ?, 'pending', ?)
                    """,
                    (
                        run_id,
                        job["id"],
                        job.get("name", ""),
                        job.get("input_b64", ""),
                        now,
                    ),
                )
            self.add_event(conn, run_id, None, {"event": "run", "run_id": run_id, "total": len(jobs)})
        return {"run_id": run_id, "total": len(jobs)}

    def add_event(self, conn: sqlite3.Connection, run_id: str, worker_id: str | None, event: dict[str, Any]) -> None:
        conn.execute(
            "INSERT INTO events (run_id, worker_id, event_json, created_at) VALUES (?, ?, ?, ?)",
            (run_id, worker_id, json.dumps(event), time.time()),
        )

    def lease(self, payload: dict[str, Any]) -> dict[str, Any]:
        worker = payload["worker"]
        worker_id = worker["worker_id"]
        warm_model = worker.get("model")  # model this worker already has resident
        batch_size = max(1, int(payload.get("batch_size") or 1))
        now = time.time()
        lease_until = now + LEASE_SECONDS
        with self._lock, self._connect() as conn:
            self._upsert_worker(conn, worker, now)
            # Work-conserving max-min fair share across users (owners). Devices in
            # use right now, per owner:
            load: dict[str, int] = {}
            for r in conn.execute(
                "SELECT r.owner AS owner, COUNT(*) AS n "
                "FROM jobs j JOIN runs r ON r.run_id = j.run_id "
                "WHERE j.status='running' AND j.lease_until >= ? GROUP BY r.owner",
                (now,),
            ):
                load[r["owner"]] = r["n"]
            # Every job that could be handed out now (queued, or a lease that
            # expired), oldest first.
            rows = conn.execute(
                """
                SELECT j.run_id, j.job_id, j.name, j.input_b64, j.updated_at, r.owner, r.model, r.config_json
                FROM jobs j
                JOIN runs r ON r.run_id = j.run_id
                WHERE r.status = 'running'
                  AND (j.status = 'pending' OR (j.status = 'running' AND j.lease_until < ?))
                ORDER BY j.updated_at, j.job_id
                """,
                (now,),
            ).fetchall()
            if not rows:
                return {"jobs": []}
            # Pick by, in order: (1) fairness — the owner using the fewest devices
            # right now; (2) model affinity — among equally-underserved owners,
            # prefer a job whose model this worker already has loaded, so it
            # doesn't reload; (3) oldest. So one user alone fills the cluster, many
            # users get a fair share, and each device tends to stay on one model
            # (reloading only when its model has no waiting work). Work-conserving
            # throughout — a device never idles while any job waits.
            def rank(row):
                return (load.get(row["owner"], 0),
                        0 if row["model"] == warm_model else 1,
                        row["updated_at"], row["job_id"])
            chosen = min(rows, key=rank)
            run_id = chosen["run_id"]
            config_json = chosen["config_json"]
            picked = [row for row in rows if row["run_id"] == run_id][:batch_size]
            jobs = []
            for row in picked:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status='running', worker_id=?, lease_until=?, attempts=attempts+1, updated_at=?
                    WHERE run_id=? AND job_id=?
                    """,
                    (worker_id, lease_until, now, row["run_id"], row["job_id"]),
                )
                jobs.append({
                    "id": row["job_id"],
                    "name": row["name"],
                    "input_b64": row["input_b64"],
                })
            return {"run_id": run_id, "config": json.loads(config_json), "jobs": jobs}

    def _upsert_worker(self, conn: sqlite3.Connection, worker: dict[str, Any], now: float) -> None:
        """Register/refresh a worker's heartbeat (last_seen)."""
        conn.execute(
            """
            INSERT INTO workers (worker_id, host, accelerator, device_id, label, last_seen)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(worker_id) DO UPDATE SET
                host=excluded.host, accelerator=excluded.accelerator,
                device_id=excluded.device_id, label=excluded.label,
                last_seen=excluded.last_seen
            """,
            (worker["worker_id"], worker["host"], worker["accelerator"],
             str(worker["device_id"]), worker["label"], now),
        )

    def heartbeat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """A liveness ping a worker sends while it's busy computing (and so not
        leasing), so the fleet never shows an active worker as offline."""
        with self._lock, self._connect() as conn:
            self._upsert_worker(conn, payload["worker"], time.time())
        return {"ok": True}

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        """Cancel a run: stop new jobs being leased (run no longer 'running')
        and mark its unfinished jobs canceled. Workers check this to abort the
        work they're currently running for the run."""
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE runs SET status='canceled', updated_at=? WHERE run_id=?", (now, run_id))
            conn.execute(
                "UPDATE jobs SET status='canceled', lease_until=NULL, updated_at=? "
                "WHERE run_id=? AND status IN ('pending','running')", (now, run_id))
            self.add_event(conn, run_id, None, {"event": "run_done", "status": "canceled", "failed": 0})
        return {"ok": True}

    def run_status(self, run_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return {"status": row["status"] if row else "missing"}

    def run_jobs(self, run_id: str) -> list[dict[str, Any]]:
        """Per-input snapshot for a run: each job's id, status and live phase.
        One cheap query regardless of event volume, so it scales to many inputs."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT job_id, status, stage FROM jobs WHERE run_id=? ORDER BY rowid",
                (run_id,),
            ).fetchall()
        return [{"id": r["job_id"], "status": r["status"], "stage": r["stage"]} for r in rows]

    def record_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = payload["run_id"]
        worker_id = payload.get("worker_id")
        event = payload["event"]
        with self._lock, self._connect() as conn:
            self.add_event(conn, run_id, worker_id, event)
            if worker_id:
                now = time.time()
                # An event is also a sign of life — keep the heartbeat fresh.
                conn.execute("UPDATE workers SET last_seen=? WHERE worker_id=?", (now, worker_id))
                # Attribute the stage to this worker's currently-running job, so
                # each input's live phase is queryable. One job per worker at a
                # time (batch_size=1), so worker_id + running pins it exactly.
                phase = _coarse_stage(event)
                if phase:
                    conn.execute(
                        "UPDATE jobs SET stage=?, updated_at=? "
                        "WHERE run_id=? AND worker_id=? AND status='running'",
                        (phase, now, run_id, worker_id))
        return {"ok": True}

    def complete_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = payload["run_id"]
        worker_id = payload["worker_id"]
        row = payload["result"]
        outputs = payload.get("outputs") or None
        status = row.get("status", "failed")
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status=?, result_json=?, outputs_json=?, error=?, lease_until=NULL, updated_at=?
                WHERE run_id=? AND job_id=?
                """,
                (
                    status,
                    json.dumps(row),
                    json.dumps(outputs) if outputs else None,
                    row.get("error"),
                    now,
                    run_id,
                    row["id"],
                ),
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
                # Only conclude a run that's still running — never overwrite a
                # 'canceled' status set by cancel_run with a late job completion.
                changed = conn.execute(
                    "UPDATE runs SET status=?, updated_at=? WHERE run_id=? AND status='running'",
                    (run_status, now, run_id),
                ).rowcount
                if changed:
                    self.add_event(conn, run_id, None, {"event": "run_done", "status": run_status, "failed": failed})
        return {"ok": True}

    def results(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return [
                json.loads(row["result_json"])
                for row in conn.execute(
                    "SELECT result_json FROM jobs WHERE run_id=? AND result_json IS NOT NULL ORDER BY updated_at",
                    (run_id,),
                )
            ]

    def job_outputs(self, run_id: str, job_id: str) -> dict[str, str]:
        """Return the output files this job produced as {name: base64_bytes}."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT outputs_json FROM jobs WHERE run_id=? AND job_id=?",
                (run_id, job_id),
            ).fetchone()
        if not row or not row["outputs_json"]:
            return {}
        try:
            return json.loads(row["outputs_json"]) or {}
        except Exception:
            return {}

    def cluster(self, stale_after: float = 20.0) -> dict[str, Any]:
        """Fleet snapshot: which workers are registered (grouped by host), how
        many are live, and run/job counts. Used for operator + platform status.

        A worker heartbeats on every lease poll (~1s when idle), so anything not
        seen within ``stale_after`` seconds is treated as gone (machine left).
        """
        now = time.time()
        with self._connect() as conn:
            workers = [dict(row) for row in conn.execute(
                "SELECT worker_id, host, accelerator, device_id, label, last_seen "
                "FROM workers ORDER BY host, device_id"
            )]
            run_rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM runs GROUP BY status"
            ).fetchall()
            job_rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
            ).fetchall()
        for w in workers:
            w["idle_s"] = round(now - float(w["last_seen"]), 1)
            w["online"] = w["idle_s"] <= stale_after
        hosts: dict[str, dict[str, Any]] = {}
        for w in workers:
            if not w["online"]:
                continue
            h = hosts.setdefault(w["host"], {"host": w["host"], "devices": 0, "accelerators": set()})
            h["devices"] += 1
            h["accelerators"].add(w["accelerator"])
        host_list = sorted(
            ({"host": h["host"], "devices": h["devices"], "accelerators": sorted(h["accelerators"])}
             for h in hosts.values()),
            key=lambda h: h["host"],
        )
        return {
            "now": now,
            "workers": workers,
            "online_workers": sum(1 for w in workers if w["online"]),
            "total_workers": len(workers),
            "hosts": host_list,
            "runs": {row["status"]: row["n"] for row in run_rows},
            "jobs": {row["status"]: row["n"] for row in job_rows},
        }

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
                    elif self.path == "/lease":
                        _json_response(self, 200, store.lease(payload))
                    elif self.path == "/events":
                        _json_response(self, 200, store.record_event(payload))
                    elif self.path == "/complete":
                        _json_response(self, 200, store.complete_job(payload))
                    elif self.path == "/heartbeat":
                        _json_response(self, 200, store.heartbeat(payload))
                    elif self.path.startswith("/runs/") and self.path.endswith("/cancel"):
                        _json_response(self, 200, store.cancel_run(self.path.split("/")[2]))
                    else:
                        _json_response(self, 404, {"error": "not found"})
                except Exception as exc:
                    _json_response(self, 500, {"error": str(exc)})

            def do_GET(self):  # noqa: N802
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path == "/cluster":
                    _json_response(self, 200, store.cluster())
                elif parsed.path == "/healthz":
                    _json_response(self, 200, {"ok": True})
                elif parsed.path.startswith("/runs/") and parsed.path.endswith("/events"):
                    run_id = parsed.path.split("/")[2]
                    query = urllib.parse.parse_qs(parsed.query)
                    after = int((query.get("after") or ["0"])[0])
                    _json_response(self, 200, store.events(run_id, after))
                elif parsed.path.startswith("/runs/") and parsed.path.endswith("/results"):
                    run_id = parsed.path.split("/")[2]
                    _json_response(self, 200, {"results": store.results(run_id)})
                elif parsed.path.startswith("/runs/") and parsed.path.endswith("/status"):
                    _json_response(self, 200, store.run_status(parsed.path.split("/")[2]))
                elif parsed.path.startswith("/runs/") and parsed.path.endswith("/jobs"):
                    _json_response(self, 200, {"jobs": store.run_jobs(parsed.path.split("/")[2])})
                elif "/jobs/" in parsed.path and parsed.path.endswith("/outputs"):
                    parts = parsed.path.split("/")
                    # /runs/<run_id>/jobs/<job_id>/outputs
                    if len(parts) == 6 and parts[1] == "runs" and parts[3] == "jobs":
                        _json_response(self, 200, {"outputs": store.job_outputs(parts[2], parts[4])})
                    else:
                        _json_response(self, 404, {"error": "not found"})
                else:
                    _json_response(self, 404, {"error": "not found"})

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.port = self.httpd.server_address[1]

    def serve_in_background(self) -> threading.Thread:
        thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        thread.start()
        return thread

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class ControllerClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
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

    def lease(self, worker: dict[str, Any], batch_size: int) -> dict[str, Any]:
        return self._request("POST", "/lease", {"worker": worker, "batch_size": batch_size})

    def event(self, run_id: str, worker_id: str, event: dict[str, Any]) -> None:
        self._request("POST", "/events", {"run_id": run_id, "worker_id": worker_id, "event": event})

    def complete(
        self,
        run_id: str,
        worker_id: str,
        row: dict[str, Any],
        event: dict[str, Any],
        outputs: dict[str, str] | None = None,
    ) -> None:
        payload = {"run_id": run_id, "worker_id": worker_id, "result": row, "event": event}
        if outputs:
            payload["outputs"] = outputs
        self._request("POST", "/complete", payload)

    def heartbeat(self, worker: dict[str, Any]) -> None:
        self._request("POST", "/heartbeat", {"worker": worker})

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        return self._request("POST", f"/runs/{run_id}/cancel", {})

    def run_status(self, run_id: str) -> str:
        try:
            return self._request("GET", f"/runs/{run_id}/status").get("status", "missing")
        except Exception:
            return "missing"

    def run_jobs(self, run_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/runs/{run_id}/jobs").get("jobs", [])

    def cluster(self) -> dict[str, Any]:
        return self._request("GET", "/cluster")

    def events(self, run_id: str, after: int) -> dict[str, Any]:
        return self._request("GET", f"/runs/{run_id}/events?after={after}")

    def results(self, run_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/runs/{run_id}/results").get("results", [])

    def job_outputs(self, run_id: str, job_id: str) -> dict[str, str]:
        return self._request("GET", f"/runs/{run_id}/jobs/{job_id}/outputs").get("outputs", {})


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
    """Build self-contained job dicts that carry the input file bytes.

    Each entry is ``{"id": <stem>, "name": <filename>, "input_b64": <base64>}``
    so a worker can run the job without sharing a filesystem with the controller.
    """
    payloads: list[dict[str, str]] = []
    for job in jobs:
        path = Path(job.path)
        payloads.append({
            "id": job.id,
            "name": path.name,
            "input_b64": base64.b64encode(path.read_bytes()).decode("ascii"),
        })
    return payloads


def worker_payload(worker: Any) -> dict[str, Any]:
    return asdict(worker) | {"label": worker.label}
