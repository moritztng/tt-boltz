"""Per-step task runner — invoked once per pipeline stage in the same process.

Replaces the original ``boltzgen.resources.main`` which used Hydra +
OmegaConf. With both dependencies removed, this is now a thin wrapper
around :func:`tt_boltz.boltzgen._config.instantiate` plus a YAML load + a
deep dict merge for CLI dotlist overrides.
"""
from __future__ import annotations

import sys
from typing import List

from tt_boltz.boltzgen._config import (
    deep_merge,
    dotlist_to_dict,
    instantiate,
    load_yaml,
    resolve_interpolations,
)
from tt_boltz.boltzgen.task.task import Task


def main(config: str, args: List[str]) -> None:
    cfg = load_yaml(config)
    if args:
        cfg = deep_merge(cfg, dotlist_to_dict(args))
    cfg = resolve_interpolations(cfg)

    task = instantiate(cfg)
    if not isinstance(task, Task):
        raise TypeError("Config must instantiate a Task subclass; got " + type(task).__name__)

    task.run(cfg)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
