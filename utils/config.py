"""Recursive YAML configuration inheritance used by every entry point."""
from copy import deepcopy
from pathlib import Path
import yaml


def merge_config(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path, _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f"Circular config include: {path}")
    seen.add(path)
    with path.open() as stream:
        config = yaml.safe_load(stream) or {}
    include = config.pop("__include__", None)
    if include:
        config = merge_config(load_config(path.parent / include, seen), config)
    return config
