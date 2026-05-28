"""YAML config helpers — minimal replacements for hydra + omegaconf.

The Hydra surface area BoltzGen actually used at inference time is small:

* ``hydra.utils.instantiate(cfg)`` — build a Task from a config dict whose
  ``_target_`` field names the class. Replaced by :func:`instantiate`.
* ``OmegaConf.load`` / ``OmegaConf.save`` — YAML I/O. Replaced by pyyaml.
* ``OmegaConf.merge`` — deep dict merge. Replaced by :func:`deep_merge`.
* ``OmegaConf.from_dotlist`` — turn ``["a.b=1", "c=x"]`` into a nested dict.
  Replaced by :func:`dotlist_to_dict`.

None of these need the rest of Hydra (composition, defaults, sweeps,
resolvers). Keeping the surface tiny means we can drop both top-level
dependencies.
"""
from __future__ import annotations

import copy
import importlib
from pathlib import Path
from typing import Any, Iterable

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def save_yaml(cfg: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)


_INTERP_RE = __import__("re").compile(r"\$\{([^}]+)\}")


def resolve_interpolations(cfg: dict) -> dict:
    """Resolve ``${key}`` / ``${a.b.c}`` references in string values.

    Each ``${path}`` is looked up by dotted path from the config root and
    substituted with the resolved value. Repeats until convergence (a single
    pass per occurrence; chains like ``${a}=${b}`` resolve in order).
    """
    def lookup(path: str, root: dict):
        node = root
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return None
        return node

    def walk(node, root):
        if isinstance(node, dict):
            return {k: walk(v, root) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v, root) for v in node]
        if isinstance(node, str) and "${" in node:
            # If the whole value is ``${path}``, substitute the typed value.
            m = _INTERP_RE.fullmatch(node)
            if m is not None:
                v = lookup(m.group(1), root)
                return v if v is not None else node
            # Otherwise, string interpolation.
            def sub(match):
                v = lookup(match.group(1), root)
                return str(v) if v is not None else match.group(0)
            return _INTERP_RE.sub(sub, node)
        return node

    out = cfg
    for _ in range(8):  # bounded to break loops; 8 is more than any real config needs
        new = walk(out, out)
        if new == out:
            break
        out = new
    return out


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge — override wins on conflicts.

    Lists are replaced (not concatenated), matching OmegaConf.merge default
    behavior.
    """
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def dotlist_to_dict(dotlist: Iterable[str]) -> dict:
    """Turn ``["a.b=1", "c=x"]`` into ``{"a": {"b": 1}, "c": "x"}``.

    Values are parsed as YAML scalars so booleans, ints, floats, and ``null``
    come back as Python natives (matching OmegaConf.from_dotlist's behavior
    close enough for our pipeline configs).
    """
    out: dict = {}
    for item in dotlist:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"Override must be key=value: {item!r}")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        keys = key.split(".")
        cursor = out
        for k in keys[:-1]:
            cursor = cursor.setdefault(k, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Override key path collides with scalar: {item!r}")
        cursor[keys[-1]] = value
    return out


def instantiate(cfg: dict) -> Any:
    """Build the object named in ``cfg['_target_']``, passing the rest as kwargs.

    Recursively descends: any nested dict that has its own ``_target_`` is
    instantiated first. Matches the only hydra.utils.instantiate behavior
    BoltzGen relies on.
    """
    if not isinstance(cfg, dict) or "_target_" not in cfg:
        # Pass-through for plain values (recursion base case).
        return cfg

    target = cfg["_target_"]
    kwargs = {k: _resolve(v) for k, v in cfg.items() if k != "_target_"}
    module_path, cls_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(**kwargs)


def _resolve(value: Any) -> Any:
    """Recursive helper for :func:`instantiate`."""
    if isinstance(value, dict):
        if "_target_" in value:
            return instantiate(value)
        return {k: _resolve(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v) for v in value]
    return value
