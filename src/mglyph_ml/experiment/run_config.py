import inspect
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, cast


def _caller_globals() -> dict[str, Any]:
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None or frame.f_back.f_back is None:
        msg = "Unable to resolve caller globals. Pass globals_dict explicitly."
        raise RuntimeError(msg)
    return frame.f_back.f_back.f_globals


class RunConfigBase:
    @classmethod
    def field_names(cls) -> set[str]:
        if not is_dataclass(cls):
            msg = f"{cls.__name__} must be a dataclass to use RunConfigBase."
            raise TypeError(msg)

        return {f.name for f in fields(cast(Any, cls))}

    @classmethod
    def clear_globals(cls, globals_dict: dict[str, Any] | None = None) -> set[str]:
        if globals_dict is None:
            globals_dict = _caller_globals()

        field_names = cls.field_names()

        # Remove stale values from prior interactive runs so Papermill overrides land cleanly.
        for name in field_names:
            globals_dict.pop(name, None)

        return field_names

    @classmethod
    def from_globals(cls, globals_dict: dict[str, Any] | None = None):
        if globals_dict is None:
            globals_dict = _caller_globals()

        field_names = cls.field_names()
        injected_values = {name: globals_dict[name] for name in field_names if name in globals_dict}
        return cls(**injected_values)


def clear_config_globals(config_cls: type, globals_dict: dict) -> set[str]:
    return config_cls.clear_globals(globals_dict)


def config_from_globals(config_cls: type, globals_dict: dict):
    return config_cls.from_globals(globals_dict)


def to_clearml_serializable(value: Any) -> Any:
    """Convert nested values into types ClearML can safely store."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: to_clearml_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_clearml_serializable(v) for v in value]
    return value
