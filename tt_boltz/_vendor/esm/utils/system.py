# Vendored from github.com/Biohub/esm @ b6b0e88 (MIT, Copyright 2026 Chan Zuckerberg
# Biohub, Inc.; see tt_boltz/_vendor/esm/LICENSE). Modified: absolute `esm.` imports
# rewritten to `tt_boltz._vendor.esm.` for in-tree vendoring.
import io
import subprocess
import typing as T
from pathlib import Path

PathLike = T.Union[str, Path]
PathOrBuffer = T.Union[PathLike, io.StringIO]


def run_subprocess_with_errorcheck(
    *popenargs,
    capture_output: bool = False,
    quiet: bool = False,
    env: dict[str, str] | None = None,
    shell: bool = False,
    executable: str | None = None,
    **kws,
) -> subprocess.CompletedProcess:
    """A command similar to subprocess.run, however the errormessage will
    contain the stderr when using this function. This makes it significantly
    easier to diagnose issues.
    """
    try:
        if capture_output:
            stdout = subprocess.PIPE
        elif quiet:
            stdout = subprocess.DEVNULL
        else:
            stdout = None

        p = subprocess.run(
            *popenargs,
            stderr=subprocess.PIPE,
            stdout=stdout,
            check=True,
            env=env,
            shell=shell,
            executable=executable,
            **kws,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Command failed with errorcode {e.returncode}." f"\n\n{e.stderr.decode()}"
        )
    return p
