"""Tiny runtime primitives used by the scheduler and CLI.

PredictionJob is one input file the user asked us to predict; WorkerSlot is
one accelerator we can run it on. Discovery and detection helpers live here
so they can be reused from both the CLI and the worker subprocess.
"""

from __future__ import annotations

import glob
import socket
from dataclasses import dataclass
from pathlib import Path


INPUT_SUFFIXES = (".fa", ".fas", ".fasta", ".yml", ".yaml")


@dataclass(frozen=True)
class PredictionJob:
    """One input target to predict."""

    id: str
    path: Path


@dataclass(frozen=True)
class WorkerSlot:
    """One execution slot that can run prediction jobs."""

    worker_id: str
    host: str
    accelerator: str
    device_id: int | str
    visible_devices: str | None = None
    logical_device_id: int = 0
    mesh_graph_descriptor: str | None = None

    @property
    def label(self) -> str:
        if self.accelerator == "tenstorrent":
            return f"{self.host}:tt{self.device_id}"
        return f"{self.host}:{self.accelerator}"


def discover_jobs(data: Path, structure_dir: Path, output_format: str, override: bool) -> list[PredictionJob]:
    """Discover runnable input files, applying resume semantics."""
    files = sorted(
        p for p in (data.glob("*") if data.is_dir() else [data])
        if p.suffix.lower() in INPUT_SUFFIXES
    )
    if not override:
        files = [p for p in files if not (structure_dir / f"{p.stem}.{output_format}").exists()]
    return [PredictionJob(id=p.stem, path=p) for p in files]


def detect_tenstorrent_devices(device_ids: str | None, num_devices: int, max_workers: int) -> list[int]:
    """Return TT device IDs selected for this run without importing ttnn."""
    all_devices = sorted(int(p.rsplit("/", 1)[-1]) for p in glob.glob("/dev/tenstorrent/[0-9]*"))
    if device_ids:
        devices = [int(d.strip()) for d in device_ids.split(",") if d.strip()]
    elif num_devices > 0:
        devices = all_devices[:num_devices]
    else:
        devices = all_devices
    return devices[:max_workers]


def build_local_workers(accelerator: str, jobs: list, devices: list[int]) -> list[WorkerSlot]:
    """Build worker slots for the local host (one per device, capped to jobs)."""
    host = socket.gethostname()
    if accelerator == "tenstorrent":
        return [
            WorkerSlot(
                worker_id=f"{host}:tt:{device}",
                host=host,
                accelerator=accelerator,
                device_id=device,
                visible_devices=str(device),
            )
            for device in devices[:len(jobs)]
        ]
    return [WorkerSlot(worker_id=f"{host}:{accelerator}:0", host=host,
                       accelerator=accelerator, device_id=0)]
