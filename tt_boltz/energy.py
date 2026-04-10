"""Lightweight Tenstorrent board energy profiling via sysfs hwmon."""

from __future__ import annotations

import csv
import importlib.util
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
    samples: int = 0
    duration_s: float | None = None
    energy_j: float | None = None
    energy_wh: float | None = None
    avg_w: float | None = None
    peak_w: float | None = None
    min_w: float | None = None
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
        metric_mode: str = "both",
    ):
        if sample_hz <= 0:
            raise ValueError("sample_hz must be > 0")
        if input_sample_hz is not None and input_sample_hz <= 0:
            raise ValueError("input_sample_hz must be > 0")
        if metric_mode not in ("both", "tdp", "input"):
            raise ValueError("metric_mode must be one of: both, tdp, input")
        self.device_id = int(device_id)
        self.sample_hz = float(sample_hz)
        self.input_sample_hz = float(input_sample_hz if input_sample_hz is not None else sample_hz)
        self.metric_mode = metric_mode
        self.enable_tdp = metric_mode in ("both", "tdp")
        self.enable_input = metric_mode in ("both", "input")
        self.interval_s = 1.0 / self.sample_hz
        self.power_path, self.sysfs_device_name, self.pci_bdf = self._resolve_paths(self.device_id, self.enable_tdp)
        self.samples: list[tuple[float, float | None, float | None]] = []  # (monotonic_time_s, tdp_power_w, input_power_w)
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
            f"tt-mgmt UMD input_power_w via helper process ({self._helper_python})"
        )

    @staticmethod
    def _resolve_paths(device_id: int, require_power_path: bool) -> tuple[Path | None, str, str]:
        """Resolve BDF (+ optional power file) using runtime device order."""
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

        if not require_power_path:
            return None, tt_dir.name, pci_bdf

        candidates = sorted((tt_dir / "device" / "hwmon").glob("hwmon*/power1_input"))
        if not candidates:
            raise RuntimeError(f"No hwmon power1_input found for device {device_id} under {tt_dir}/device/hwmon")
        return candidates[0], tt_dir.name, pci_bdf

    def _read_power_w(self) -> float:
        if self.power_path is None:
            raise RuntimeError("TDP channel is disabled")
        raw = self.power_path.read_text().strip()
        return int(raw) / 1_000_000.0  # microwatts -> watts

    def _resolve_helper_python(self) -> Path:
        """Pick python executable for isolated tt-mgmt reads."""
        env_override = os.environ.get("TT_POWER_HELPER_PYTHON")
        if env_override:
            return Path(env_override).expanduser().resolve()
        return Path(sys.executable)

    def _start_input_power_helper(self) -> subprocess.Popen[str]:
        """Start helper process that continuously prints INPUT_POWER samples."""
        if not self._helper_python.exists():
            raise RuntimeError(f"helper python not found: {self._helper_python}")
        # Keep power measurement optional: only users of --report-energy need tt-mgmt.
        if self._helper_python == Path(sys.executable) and importlib.util.find_spec("tt_mgmt") is None:
            raise RuntimeError(
                "tt-mgmt is required for --report-energy input_power_w.\n"
                "Install once:\n"
                "  git clone --recursive https://github.com/aperezvicente-TT/tt-mgmt.git\n"
                "  pip install -e ./tt-mgmt"
            )
        script = (
            "import sys\n"
            "import time\n"
            "target_bdf=sys.argv[1].lower()\n"
            "hz=float(sys.argv[2])\n"
            "dt=1.0/hz\n"
            "from tt_mgmt.api import connect\n"
            "c=connect(embedded=True, backend='umd')\n"
            "devs=c.device_list()\n"
            "idx=None\n"
            "for i,d in enumerate(devs):\n"
            "  if str(d.get('pci_bdf','')).lower()==target_bdf:\n"
            "    idx=i\n"
            "    break\n"
            "if idx is None:\n"
            "  raise RuntimeError(f'pci_bdf {target_bdf} not found in tt-mgmt device list')\n"
            "while True:\n"
            "  d=c.update_telemetry(idx)\n"
            "  t=d.get('telemetry',{})\n"
            "  p=t.get('input_power_w')\n"
            "  if p is None:\n"
            "    raise RuntimeError('input_power_w unavailable from tt-mgmt telemetry')\n"
            "  print(f'{float(p)}', flush=True)\n"
            "  time.sleep(dt)\n"
        )
        proc = subprocess.Popen(
            [str(self._helper_python), "-u", "-c", script, self.pci_bdf, str(self.input_sample_hz)],
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
                        if self._stop.is_set():
                            break
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
                power_w = self._read_power_w() if self.enable_tdp else None
                with self._input_power_lock:
                    input_power_w = self._input_power_latest if self.enable_input else None
                if power_w is not None or input_power_w is not None:
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
        if self.enable_input:
            self._input_thread = threading.Thread(target=self._input_loop, name="tt-input-power-profiler", daemon=True)
            self._input_thread.start()
        else:
            self._input_thread = None
        self._thread = threading.Thread(target=self._loop, name="tt-power-profiler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        proc = self._input_proc
        if self.enable_input and proc is not None and proc.poll() is None:
            proc.terminate()
        if self._input_thread is not None:
            self._input_thread.join(timeout=2.0)

    def summarize(self) -> EnergySummary:
        tdp_pts = [(t, p) for t, p, _ in self.samples if p is not None]
        if self.enable_tdp and tdp_pts:
            duration = max(0.0, tdp_pts[-1][0] - tdp_pts[0][0])
            energy_j = 0.0
            for i in range(1, len(tdp_pts)):
                t0, p0 = tdp_pts[i - 1]
                t1, p1 = tdp_pts[i]
                dt = max(0.0, t1 - t0)
                energy_j += 0.5 * (p0 + p1) * dt
            powers = [p for _, p in tdp_pts]
            samples = len(tdp_pts)
            avg_w = sum(powers) / len(powers)
            peak_w = max(powers)
            min_w = min(powers)
            energy_wh = energy_j / 3600.0
        else:
            samples = 0
            duration = None
            energy_j = None
            energy_wh = None
            avg_w = None
            peak_w = None
            min_w = None

        input_pts = self.input_samples if self.enable_input else []
        if input_pts:
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
            input_samples = 0
            input_duration_s = None
            input_energy_j = None
            input_energy_wh = None
            input_avg_w = None
            input_peak_w = None
            input_min_w = None

        return EnergySummary(
            samples=samples,
            duration_s=duration,
            energy_j=energy_j,
            energy_wh=energy_wh,
            avg_w=avg_w,
            peak_w=peak_w,
            min_w=min_w,
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
        fields = ["t_rel_s"]
        if self.enable_tdp:
            fields.append("power_w")
        if self.enable_input:
            fields.append("input_power_w")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(fields)
            if not self.samples:
                return
            t0 = self.samples[0][0]
            for t, p, ip in self.samples:
                row = [f"{(t - t0):.6f}"]
                if self.enable_tdp:
                    row.append("" if p is None else f"{p:.6f}")
                if self.enable_input:
                    row.append("" if ip is None else f"{ip:.6f}")
                w.writerow(row)

    def write_plot(self, path: Path, title: str = "Power vs Time") -> bool:
        """Return True if plot written, False if matplotlib unavailable."""
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return False
        if not self.samples:
            return False
        t0 = self.samples[0][0]
        xs = [t - t0 for t, p, _ in self.samples if p is not None]
        ys = [p for _, p, _ in self.samples if p is not None]
        ixs = [t - t0 for t, _, ip in self.samples if ip is not None]
        iys = [ip for _, _, ip in self.samples if ip is not None]
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        if ys:
            ax.plot(xs, ys, linewidth=1.5, label="tdp_power_w")
        if iys:
            ax.plot(ixs, iys, linewidth=1.5, label="input_power_w", alpha=0.9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power (W)")
        ax.set_title(title)
        if ys or iys:
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
        if self.power_path is None:
            return f"{self.sysfs_device_name} ({self.pci_bdf})"
        return f"{self.sysfs_device_name} ({self.pci_bdf}) @ {self.power_path}"

    @property
    def input_power_source(self) -> str | None:
        return self._input_power_source

    @property
    def input_power_note(self) -> str | None:
        return self._input_power_note

