"""This file contains some helper functions that help you create a set of commands to run."""

from itertools import product
from pathlib import Path
import shlex


def format_overrides(overrides: dict[str, object], separator: str) -> str:
    return separator.join(f"{key}={value}" for key, value in overrides.items())


def format_run_title(
    base_name: str,
    overrides: dict[str, object],
    parameter_name_aliases: dict[str, str] | None = None,
) -> str:
    parameter_name_aliases = parameter_name_aliases or {}
    title_parts = [f"{parameter_name_aliases.get(key, key)}={value}" for key, value in overrides.items()]
    return f"{base_name}: {','.join(title_parts)}"


def build_convert_notebook_command(
    notebook_path: Path,
    base_name: str,
    overrides: dict[str, object],
    parameter_name_aliases: dict[str, str] | None = None,
) -> str:
    run_title = format_run_title(base_name, overrides, parameter_name_aliases)
    serialized_overrides = format_overrides(overrides, separator=" ")
    return (
        f"./convert-notebook.sh {shlex.quote(str(notebook_path))} "
        f"--name={shlex.quote(run_title)} "
        f"{serialized_overrides}"
    )


def build_convert_notebook_commands(
    notebook_path: Path,
    base_name: str,
    override_grid: list[dict[str, object]],
    parameter_name_aliases: dict[str, str] | None = None,
) -> str:
    commands = [
        build_convert_notebook_command(
            notebook_path,
            base_name,
            overrides,
            parameter_name_aliases,
        )
        for overrides in override_grid
    ]
    return "\n".join(commands)


def build_parameter_grid(**parameter_options: list[object]) -> list[dict[str, object]]:
    keys = list(parameter_options.keys())
    value_lists = [parameter_options[key] for key in keys]
    return [dict(zip(keys, combination)) for combination in product(*value_lists)]
