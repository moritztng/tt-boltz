"""Lightweight Tenstorrent board energy profiling via sysfs hwmon."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from io import TextIOWrapper
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
    input_samples: int = 0
    input_duration_s: float | None = None
    input_energy_j: float | None = None
    input_energy_wh: float | None = None
    input_avg_w: float | None = None
    input_peak_w: float | None = None
    input_min_w: float | None = None


DEFAULT_ENERGY_SAMPLE_HZ = 20.0


class SysfsPowerProfiler:
    """Sample one Tenstorrent device power from sysfs and integrate energy."""

    def __init__(
        self,
        device_id: int,
        sample_hz: float = DEFAULT_ENERGY_SAMPLE_HZ,
        input_sample_hz: float | None = None,
    ):
        if sample_hz <= 0:
            raise ValueError("sample_hz must be > 0")
        if input_sample_hz is not None and input_sample_hz <= 0:
            raise ValueError("input_sample_hz must be > 0")
        self.device_id = int(device_id)
        self.sample_hz = float(sample_hz)
        self.input_sample_hz = float(input_sample_hz if input_sample_hz is not None else sample_hz)
        self.interval_s = 1.0 / self.sample_hz
        self.power_path, self.sysfs_device_name, self.pci_bdf = self._resolve_power_path(self.device_id)
        self.samples: list[tuple[float, float, float | None]] = []  # (monotonic_time_s, tdp_power_w, input_power_w)
        self.input_samples: list[tuple[float, float]] = []  # (monotonic_time_s, input_power_w)
        self._thread: threading.Thread | None = None
        self._input_thread: threading.Thread | None = None
        self._input_proc: subprocess.Popen[str] | None = None
        self._stop = threading.Event()
        self._error: str | None = None
        self._input_power_note: str | None = None
        self._input_power_latest: float | None = None
        self._input_power_lock = threading.Lock()
        self._helper_python = self._resolve_helper_python()
        self._input_power_source = (
            f"umd TelemetryTag 54 (INPUT_POWER) via helper process ({self._helper_python})"
        )

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

    def _resolve_helper_python(self) -> Path:
        """Pick python executable for isolated tt_umd reads."""
        env_override = os.environ.get("TT_UMD_PYTHON")
        if env_override:
            return Path(env_override).expanduser()
        return Path(sys.executable)

    def _start_input_power_helper(self) -> subprocess.Popen[str]:
        """Start helper process that continuously prints INPUT_POWER samples."""
        if not self._helper_python.exists():
            raise RuntimeError(f"helper python not found: {self._helper_python}")
        script = (
            "import sys\n"
            "import time\n"
            "from tt_umd import TopologyDiscovery, TopologyDiscoveryOptions\n"
            "dev=int(sys.argv[1])\n"
            "hz=float(sys.argv[2])\n"
            "dt=1.0/hz\n"
            "A=TopologyDiscoveryOptions.Action\n"
            "o=TopologyDiscoveryOptions();o.eth_fw_mismatch_action=A.IGNORE;o.eth_fw_heartbeat_failure=A.IGNORE;"
            "o.cmfw_mismatch_action=A.IGNORE;o.unexpected_routing_firmware_config=A.IGNORE\n"
            "_,ds=TopologyDiscovery.discover(o)\n"
            "if dev not in ds:\n"
            "  raise RuntimeError(f'device_id {dev} not found')\n"
            "r=ds[dev].get_arc_telemetry_reader()\n"
            "if not r.is_entry_available(54):\n"
            "  raise RuntimeError('INPUT_POWER tag 54 unavailable')\n"
            "while True:\n"
            "  print(r.read_entry(54), flush=True)\n"
            "  time.sleep(dt)\n"
        )
        proc = subprocess.Popen(
            [str(self._helper_python), "-u", "-c", script, str(self.device_id), str(self.input_sample_hz)],
            text=True,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            bufsize=1,
        )
        return proc

    def _input_loop(self) -> None:
        try:
            proc = self._start_input_power_helper()
        except Exception as e:
            if self._input_power_note is None:
                self._input_power_note = f"input_power unavailable: {e}"
            return

        self._input_proc = proc
        stdout: TextIOWrapper | None = proc.stdout
        if stdout is None:
            if self._input_power_note is None:
                self._input_power_note = "input_power unavailable: helper stdout not available"
            return

        try:
            while not self._stop.is_set():
                line = stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        if self._input_power_note is None:
                            self._input_power_note = f"input_power unavailable: helper exited with code {proc.returncode}"
                        break
                    continue
                now = time.monotonic()
                try:
                    value = float(line.strip())
                except ValueError:
                    continue
                with self._input_power_lock:
                    self._input_power_latest = value
                    self.input_samples.append((now, value))
        finally:
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except Exception:
                proc.kill()
            self._input_proc = None

    def _loop(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            try:
                power_w = self._read_power_w()
                with self._input_power_lock:
                    input_power_w = self._input_power_latest
                self.samples.append((now, power_w, input_power_w))
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
        self._input_power_note = None
        self.input_samples.clear()
        with self._input_power_lock:
            self._input_power_latest = None
        self._input_thread = threading.Thread(target=self._input_loop, name="tt-input-power-profiler", daemon=True)
        self._input_thread.start()
        self._thread = threading.Thread(target=self._loop, name="tt-power-profiler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        proc = self._input_proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
        if self._input_thread is not None:
            self._input_thread.join(timeout=2.0)

    def summarize(self) -> EnergySummary:
        if not self.samples:
            return EnergySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        pts = self.samples
        duration = max(0.0, pts[-1][0] - pts[0][0])
        # Trapezoidal integration over monotonic time.
        energy_j = 0.0
        for i in range(1, len(pts)):
            t0, p0, _ = pts[i - 1]
            t1, p1, _ = pts[i]
            dt = max(0.0, t1 - t0)
            energy_j += 0.5 * (p0 + p1) * dt
        powers = [p for _, p, _ in pts]
        avg_w = sum(powers) / len(powers)

        input_pts = self.input_samples
        if len(input_pts) >= 2:
            input_duration_s = max(0.0, input_pts[-1][0] - input_pts[0][0])
            input_energy_j = 0.0
            for i in range(1, len(input_pts)):
                t0, p0 = input_pts[i - 1]
                t1, p1 = input_pts[i]
                dt = max(0.0, t1 - t0)
                input_energy_j += 0.5 * (p0 + p1) * dt
            input_powers = [p for _, p in input_pts]
            input_avg_w = sum(input_powers) / len(input_powers)
            input_peak_w = max(input_powers)
            input_min_w = min(input_powers)
            input_energy_wh = input_energy_j / 3600.0
            input_samples = len(input_pts)
        else:
            input_duration_s = None
            input_energy_j = None
            input_energy_wh = None
            input_avg_w = None
            input_peak_w = None
            input_min_w = None
            input_samples = len(input_pts)

        return EnergySummary(
            samples=len(pts),
            duration_s=duration,
            energy_j=energy_j,
            energy_wh=energy_j / 3600.0,
            avg_w=avg_w,
            peak_w=max(powers),
            min_w=min(powers),
            input_samples=input_samples,
            input_duration_s=input_duration_s,
            input_energy_j=input_energy_j,
            input_energy_wh=input_energy_wh,
            input_avg_w=input_avg_w,
            input_peak_w=input_peak_w,
            input_min_w=input_min_w,
        )

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.samples:
            path.write_text("t_rel_s,power_w,input_power_w\n")
            return
        t0 = self.samples[0][0]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_rel_s", "power_w", "input_power_w"])
            for t, p, ip in self.samples:
                w.writerow([f"{(t - t0):.6f}", f"{p:.6f}", "" if ip is None else f"{ip:.6f}"])

    def write_plot(self, path: Path, title: str = "Power vs Time") -> bool:
        """Return True if plot written, False if matplotlib unavailable."""
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return False
        if not self.samples:
            return False
        t0 = self.samples[0][0]
        xs = [t - t0 for t, _, _ in self.samples]
        ys = [p for _, p, _ in self.samples]
        ixs = [t - t0 for t, _, ip in self.samples if ip is not None]
        iys = [ip for _, _, ip in self.samples if ip is not None]
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.plot(xs, ys, linewidth=1.5, label="tdp_power_w")
        if iys:
            ax.plot(ixs, iys, linewidth=1.5, label="input_power_w", alpha=0.9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power (W)")
        ax.set_title(title)
        if iys:
            ax.legend()
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

    @property
    def input_power_source(self) -> str | None:
        return self._input_power_source

    @property
    def input_power_note(self) -> str | None:
        return self._input_power_note

