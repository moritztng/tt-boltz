import re
import shutil
import subprocess
from pathlib import Path

import ttnn


TTNN_ROOT = Path(ttnn.__file__).resolve().parent
RUNTIME_SFPI = TTNN_ROOT / "runtime" / "sfpi"
SYSTEM_SFPI = Path("/opt/tenstorrent/sfpi")
KERNEL_CACHE = Path.home() / ".cache" / "tt-metal-cache"


def _expected_sfpi_version() -> str:
    version_file = TTNN_ROOT / "tt_metal" / "sfpi-version"
    match = re.search(r"^sfpi_version=['\"]?([^'\"\n]+)", version_file.read_text(), re.MULTILINE)
    if not match:
        raise RuntimeError(f"Unable to parse sfpi_version from {version_file}")
    return match.group(1)


def _install_sfpi(version: str) -> None:
    if shutil.which("apt-get"):
        subprocess.check_call(["sudo", "apt-get", "install", "-y", "--allow-downgrades", f"sfpi={version}"])
        return

    if shutil.which("dnf"):
        subprocess.check_call(["sudo", "dnf", "install", "-y", f"sfpi-{version}"])
        return

    raise RuntimeError(f"Unsupported package manager. Install sfpi {version} manually.")


def _use_system_sfpi() -> None:
    if not RUNTIME_SFPI.exists() and not RUNTIME_SFPI.is_symlink():
        return

    if not RUNTIME_SFPI.is_symlink():
        print(f"Keeping bundled SFPI at {RUNTIME_SFPI}")
        return

    if RUNTIME_SFPI.resolve(strict=False) == SYSTEM_SFPI:
        return

    RUNTIME_SFPI.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_SFPI.unlink(missing_ok=True)
    RUNTIME_SFPI.symlink_to(SYSTEM_SFPI, target_is_directory=True)
    print(f"Linked {RUNTIME_SFPI} -> {SYSTEM_SFPI}")


def main() -> None:
    version = _expected_sfpi_version()
    print(f"Installing SFPI {version} for ttnn at {TTNN_ROOT}")
    _install_sfpi(version)
    _use_system_sfpi()

    shutil.rmtree(KERNEL_CACHE, ignore_errors=True)
    print(f"Cleared {KERNEL_CACHE}")


if __name__ == "__main__":
    main()
