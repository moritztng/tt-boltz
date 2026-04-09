"""Lightweight Tenstorrent board energy profiling via sysfs hwmon."""

from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EnergySummary:
    samples: int
    duration_s: float
    energy_j: float
    energy_wh: float
    avg_w: float
    peak_w: float
    min_w: float


DEFAULT_ENERGY_SAMPLE_HZ = 20.0


class SysfsPowerProfiler:
    """Sample one Tenstorrent device power from sysfs and integrate energy."""

    def __init__(self, device_id: int, sample_hz: float = 10.0):
        if sample_hz <= 0:
            raise ValueError("sample_hz must be > 0")
        self.device_id = int(device_id)
        self.sample_hz = float(sample_hz)
        self.interval_s = 1.0 / self.sample_hz
        self.power_path, self.sysfs_device_name, self.pci_bdf = self._resolve_power_path(self.device_id)
        self.samples: list[tuple[float, float]] = []  # (monotonic_time_s, power_w)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: str | None = None

    @staticmethod
    def _resolve_power_path(device_id: int) -> tuple[Path, str, str]:
        """Resolve power file using TT runtime device order (sorted PCI BDF)."""
        entries = []
        for entry in sorted(Path("/sys/class/tenstorrent").glob("tenstorrent!*")):
            try:
                pci_bdf = entry.resolve().parent.parent.name
            except Exception:
                continue
            entries.append((pci_bdf, entry))

        if not entries:
            raise RuntimeError("No /sys/class/tenstorrent/tenstorrent!* entries found")

        # tt-boltz --device_ids uses TT runtime ordering, which follows sorted PCI BDF.
        entries.sort(key=lambda x: x[0])
        if device_id < 0 or device_id >= len(entries):
            raise RuntimeError(
                f"Runtime device_id {device_id} out of range; found {len(entries)} Tenstorrent devices"
            )
        pci_bdf, tt_dir = entries[device_id]

        candidates = sorted((tt_dir / "device" / "hwmon").glob("hwmon*/power1_input"))
        if not candidates:
            raise RuntimeError(f"No hwmon power1_input found for device {device_id} under {tt_dir}/device/hwmon")
        return candidates[0], tt_dir.name, pci_bdf

    def _read_power_w(self) -> float:
        raw = self.power_path.read_text().strip()
        return int(raw) / 1_000_000.0  # microwatts -> watts

    def _loop(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            try:
                power_w = self._read_power_w()
                self.samples.append((now, power_w))
            except Exception as e:  # best effort sampling; keep first error
                if self._error is None:
                    self._error = str(e)
            next_tick += self.interval_s
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                self._stop.wait(sleep_s)

    def start(self) -> None:
        self._stop.clear()
        self.samples.clear()
        self._error = None
        self._thread = threading.Thread(target=self._loop, name="tt-power-profiler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summarize(self) -> EnergySummary:
        if not self.samples:
            return EnergySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        pts = self.samples
        duration = max(0.0, pts[-1][0] - pts[0][0])
        # Trapezoidal integration over monotonic time.
        energy_j = 0.0
        for i in range(1, len(pts)):
            t0, p0 = pts[i - 1]
            t1, p1 = pts[i]
            dt = max(0.0, t1 - t0)
            energy_j += 0.5 * (p0 + p1) * dt
        powers = [p for _, p in pts]
        avg_w = sum(powers) / len(powers)
        return EnergySummary(
            samples=len(pts),
            duration_s=duration,
            energy_j=energy_j,
            energy_wh=energy_j / 3600.0,
            avg_w=avg_w,
            peak_w=max(powers),
            min_w=min(powers),
        )

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.samples:
            path.write_text("t_rel_s,power_w\n")
            return
        t0 = self.samples[0][0]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_rel_s", "power_w"])
            for t, p in self.samples:
                w.writerow([f"{(t - t0):.6f}", f"{p:.6f}"])

    def write_plot(self, path: Path, title: str = "Power vs Time") -> bool:
        """Return True if plot written, False if matplotlib unavailable."""
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return False
        if not self.samples:
            return False
        t0 = self.samples[0][0]
        xs = [t - t0 for t, _ in self.samples]
        ys = [p for _, p in self.samples]
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.plot(xs, ys, linewidth=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power (W)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return True

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def source(self) -> str:
        return f"{self.sysfs_device_name} ({self.pci_bdf}) @ {self.power_path}"

