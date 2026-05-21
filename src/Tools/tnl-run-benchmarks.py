#!/usr/bin/env -S uv run --script
# SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pydantic>=2",
#   "PyYAML",
# ]
# ///

"""Run TNL benchmarks from a YAML or TOML configuration file.

Replaces scattered shell wrapper scripts (run-tnl-benchmark-*) with a
unified, configurable runner.

Benchmark discovery
-------------------
Benchmarks are **discovered** from the ``bin_dir`` directory: every
executable whose filename starts with ``benchmark_prefix`` (default
``tnl-benchmark-``) is treated as a benchmark. The benchmark *name* is
the filename with the prefix stripped (e.g. ``tnl-benchmark-sort-cuda``
→ name ``sort-cuda``).

During discovery, each executable's ``--help`` output is parsed to
determine which CLI parameters are *scalar* (``string``, ``integer``,
``real``, ``bool``) and which are *list-of* (``list of string``,
``list of integer``, etc.).

The ``benchmarks`` config section is optional and only used to customize
invocation (extra params, log file, command prefix, etc.). Discovered
benchmarks without a config entry run with common defaults only — one
invocation, no extra options.

Config entries that reference benchmarks not found on disk are skipped
with a warning.

Param dispatch rules
--------------------
All params live in the single ``params`` dict. The dispatch strategy for
list values depends on the parameter's CLI type (auto-detected from
``--help``):

- **Scalar params** (``string``, ``integer``, ``real``, ``bool``): a
  list value produces a **cartesian product loop** — one invocation per
  value, combined with all other looping params.
- **List-of params** (``list of string``, ``list of integer``, etc.):
  a list value is passed as **space-separated** ``--key val1 val2 ...``
  in a single invocation (no looping).

Scalar values are always passed as-is.

Params whose CLI type is unknown (not seen in ``--help``) are treated as
scalar by default.

Device parameter convention
--------------------------
TNL benchmarks handle device dispatch internally (``host``, ``cuda``,
``hip``, ``all``).  Set ``device: all`` (scalar) in the config rather
than looping over individual devices — the benchmark's
``resolveDevice`` function will run each supported device.

Auto-managed options
--------------------
``--output-mode`` and ``--log-file`` are injected automatically:

- ``--output-mode append`` is always passed (multi-invocation benchmarks
  must append after the first run).
- When ``common.output_mode`` is ``overwrite``, the log file is deleted
  before the first invocation.
- ``--log-file`` is injected from the benchmark's ``log_file`` field.

Do **not** add ``output-mode`` or ``log-file`` to ``params`` — they will
be stripped silently.

Environment variables
---------------------
Both ``common`` and per-benchmark entries accept an ``env`` dict of
environment variables.  Per-benchmark values override common ones.
Variables are **added** to the current environment (not a clean slate),
so you only need to set the ones you want to change::

    common:
      env:
        CUDA_VISIBLE_DEVICES: "0"

    benchmarks:
      sort-cuda:
        env:
          CUDA_VISIBLE_DEVICES: "1"   # overrides common

Command prefix / MPI launcher
-----------------------------
The ``command_prefix`` field on each benchmark entry prepends arbitrary
words to the command line (split on whitespace).  This serves double
duty as an MPI launcher — just set it to ``mpirun -np 4`` (or
``srun -n 4``, etc.).  Combine with taskset if needed::

    benchmarks:
      linear-solvers-cuda:
        command_prefix: "mpirun -np 4 taskset -c 0,2,4,6"

YAML boolean gotcha
-------------------
PyYAML (YAML 1.1) parses bare ``yes``, ``no``, ``on``, ``off``, ``true``,
``false`` as Python booleans. The script converts booleans to ``true``/``false``
for TNL CLI compatibility (TNL accepts both ``yes``/``no`` and ``true``/``false``
for bool-typed parameters, but ``true``/``false`` is conventional).
If you need the literal string ``"on"`` or ``"off"``, quote it:
``access-type: "on"``.

Usage
-----
::

    uv run tnl-run-benchmarks.py -c benchmarks.yaml
    uv run tnl-run-benchmarks.py -c benchmarks.yaml -n               # dry-run
    uv run tnl-run-benchmarks.py -c benchmarks.yaml -b sort          # select by name
    uv run tnl-run-benchmarks.py -c benchmarks.yaml --include '*-cuda'  # glob filter
    uv run tnl-run-benchmarks.py -c benchmarks.yaml --exclude '*-host'  # glob filter
    uv run tnl-run-benchmarks.py -c benchmarks.yaml --list           # list benchmarks
"""

import argparse
import fnmatch
import itertools
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

# ── Config schema ────────────────────────────────────────────────────────


class CommonConfig(BaseModel):
    """Options shared across all benchmarks."""

    model_config = {"extra": "forbid"}

    # directory containing benchmark executables
    bin_dir: Path = Path("./build/bin")
    # filename prefix for auto-discovery
    benchmark_prefix: str = "tnl-benchmark-"
    # log file write mode
    output_mode: Literal["append", "overwrite"] = "append"
    # verbosity level passed to benchmarks
    verbose: int = Field(default=0, ge=0)
    # number of measurement loops
    loops: int = Field(default=10, ge=1)
    # default params for all benchmarks
    params: dict[str, Any] = Field(default_factory=dict)
    # environment variables set for all benchmarks
    env: dict[str, str] = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    """Per-benchmark configuration (user-supplied).

    If ``executable`` is omitted, it is derived from the benchmark name:
    ``{benchmark_prefix}{name}``.
    """

    model_config = {"extra": "forbid"}

    # override auto-discovered executable name
    executable: str | None = None
    # path to benchmark output log
    log_file: str | None = None
    # e.g. ``taskset -c 0,2,4,6``; for MPI use ``mpirun -np 4``
    # (or combined: ``mpirun -np 4 taskset -c 0,2,4,6``)
    command_prefix: str | None = None
    # params (dispatch determined by CLI type introspection)
    params: dict[str, Any] = Field(default_factory=dict)
    # environment variables (merged with common.env)
    env: dict[str, str] = Field(default_factory=dict)


class ResolvedBenchmark(BenchmarkConfig):
    """Merged benchmark state after discovery + config overlay.

    Created internally by the script — never deserialized from user config.
    """

    model_config = {"frozen": True}

    # resolved executable filename (never None)
    final_executable: str
    # param names detected as "list of *" from --help
    list_of_params: frozenset[str] = frozenset()
    # CLI params marked REQUIRED
    required_params: list[str] = Field(default_factory=list)
    # if set, benchmark is skipped with this message
    skip_reason: str | None = None


class Config(BaseModel):
    """Top-level configuration."""

    model_config = {"extra": "forbid"}

    common: CommonConfig = Field(default_factory=CommonConfig)
    benchmarks: dict[str, BenchmarkConfig] = Field(default_factory=dict)


# ── Config loading ──────────────────────────────────────────────────────


def load_config(path: Path) -> Config:
    suffix = path.suffix.lower()
    match suffix:
        case ".yaml" | ".yml":
            data = yaml.safe_load(path.read_text())
        case ".toml":
            data = tomllib.loads(path.read_text())
        case _:
            raise SystemExit(
                f"Unsupported config format: {suffix} (expected .yaml, .yml, or .toml)"
            )
    return Config.model_validate(data)


# ── Benchmark introspection ─────────────────────────────────────────────


_REQUIRED_RE = re.compile(r"--(\S+).*\*+\s*REQUIRED\s*\*+")

_SCALAR_TYPES = r"string|integer|unsigned\s+integer|real|bool"

_PARAM_TYPE_RE = re.compile(
    rf"^\s+--(\S+)\s+(list\s+of\s+(?:{_SCALAR_TYPES})|{_SCALAR_TYPES})\b",
    re.MULTILINE,
)


def _probe_help(exe_path: Path) -> tuple[list[str], frozenset[str]]:
    """Run ``exe_path --help`` and parse REQUIRED and list-of params.

    Returns (required_params, list_of_params) where *required_params* is
    a list of param names marked ``*** REQUIRED ***`` and *list_of_params*
    is a frozenset of param names whose CLI type is any ``list of *`` variant.
    """
    try:
        proc = subprocess.run(
            [str(exe_path), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return [], frozenset()
    combined = proc.stdout + proc.stderr

    required = _REQUIRED_RE.findall(combined)

    list_of: set[str] = set()
    for m in _PARAM_TYPE_RE.finditer(combined):
        name, ptype = m.group(1), m.group(2)
        if ptype.split()[0] == "list" and ptype.split()[1] == "of":
            list_of.add(name)

    return required, frozenset(list_of)


# ── Benchmark discovery ─────────────────────────────────────────────────


def discover_benchmarks(
    bin_dir: Path,
    prefix: str,
) -> dict[str, Path]:
    """Scan *bin_dir* for executables matching *prefix*.

    Returns a sorted dict mapping benchmark name (filename with *prefix*
    stripped) to the executable path. No introspection is performed.
    """
    if not bin_dir.is_dir():
        print(f"WARNING: bin_dir does not exist: {bin_dir}", file=sys.stderr)
        return {}

    discovered: dict[str, Path] = {}
    for entry in sorted(bin_dir.iterdir()):
        if not entry.name.startswith(prefix):
            continue
        if not entry.is_file():
            continue
        if not os.access(entry, os.X_OK):
            continue
        name = entry.name.removeprefix(prefix)
        discovered[name] = entry

    return discovered


def introspect_benchmarks(
    entries: dict[str, Path],
) -> dict[str, ResolvedBenchmark]:
    """Run ``--help`` on each executable and return resolved benchmarks."""
    resolved: dict[str, ResolvedBenchmark] = {}
    for name, exe_path in entries.items():
        required, list_of = _probe_help(exe_path)
        resolved[name] = ResolvedBenchmark(
            final_executable=exe_path.name,
            list_of_params=list_of,
            required_params=required,
        )
    return resolved


def resolve_benchmarks(
    config: Config,
    selected_names: set[str] | None = None,
) -> dict[str, ResolvedBenchmark]:
    """Merge discovered benchmarks with config entries.

    - Discovered benchmarks get their config overlay (params, env,
      log_file, command_prefix) if a matching entry exists.
    - Config-only entries (not on disk) produce a warning and are skipped.
    - If *selected_names* is given, only those benchmarks are introspected
      with ``--help``; others are skipped entirely.
    """
    discovered = discover_benchmarks(
        config.common.bin_dir, config.common.benchmark_prefix
    )

    # If filters were pre-applied, only introspect the selected subset
    if selected_names is not None:
        to_introspect = {n: p for n, p in discovered.items() if n in selected_names}
    else:
        to_introspect = discovered

    introspected = introspect_benchmarks(to_introspect)

    resolved: dict[str, ResolvedBenchmark] = {}

    for name, disc_cfg in introspected.items():
        if name in config.benchmarks:
            user_cfg = config.benchmarks[name]
            merged = ResolvedBenchmark(
                final_executable=user_cfg.executable or disc_cfg.final_executable,
                log_file=user_cfg.log_file,
                command_prefix=user_cfg.command_prefix,
                params=disc_cfg.params | user_cfg.params,
                env=disc_cfg.env | user_cfg.env,
                list_of_params=disc_cfg.list_of_params,
                required_params=disc_cfg.required_params,
            )
        else:
            merged = disc_cfg

        all_configured_keys = set(merged.params)
        missing = [p for p in merged.required_params if p not in all_configured_keys]
        if missing:
            if name in config.benchmarks:
                raise SystemExit(
                    f"ERROR: [{name}] requires --{', --'.join(missing)} "
                    f"but no value provided in config"
                )
            merged = merged.model_copy(
                update={
                    "skip_reason": (
                        f"requires --{', --'.join(missing)} "
                        f"but no value provided in config"
                    ),
                }
            )

        resolved[name] = merged

    for name, user_cfg in config.benchmarks.items():
        if name not in discovered:
            exe_name = user_cfg.executable or f"{config.common.benchmark_prefix}{name}"
            exe_path = config.common.bin_dir / exe_name
            print(
                f"WARNING: [{name}] configured but not found "
                f"in {config.common.bin_dir} "
                f"(expected {exe_path}), skipping",
                file=sys.stderr,
            )

    return resolved


# ── Value formatting ────────────────────────────────────────────────────


def format_value(value: Any) -> str:
    """Format a Python value as a CLI string argument for TNL benchmarks."""
    match value:
        case None:
            raise ValueError("Parameter values cannot be None")
        case bool():
            return "true" if value else "false"
        case _:
            return str(value)


# ── Invocation builder ──────────────────────────────────────────────────

_AUTO_MANAGED_KEYS = frozenset({"output-mode", "log-file"})


def build_invocations(
    name: str,
    benchmark: ResolvedBenchmark,
    common: CommonConfig,
) -> tuple[list[str], list[list[str]], list[dict[str, str]]]:
    """Build all command invocations for a benchmark.

    Returns:
        (loop_keys, invocations, combo_labels) where *loop_keys* lists the
        param names that vary across invocations and *combo_labels* maps each
        invocation index to ``{key: value}`` pairs for the varying params.
    """
    exe_name = benchmark.final_executable or f"{common.benchmark_prefix}{name}"
    executable = str(common.bin_dir / exe_name)

    base_params: dict[str, Any] = {
        "verbose": common.verbose,
        "loops": common.loops,
    }

    merged = base_params | common.params | benchmark.params

    for key in _AUTO_MANAGED_KEYS:
        merged.pop(key, None)

    scalar_params: dict[str, str] = {}
    list_of_params: dict[str, list[str]] = {}
    loop_keys: list[str] = []
    loop_values: list[list[str]] = []

    for key, value in merged.items():
        if isinstance(value, list) and key not in benchmark.list_of_params:
            loop_keys.append(key)
            loop_values.append([format_value(v) for v in value])
        elif isinstance(value, list) and key in benchmark.list_of_params:
            list_of_params[key] = [format_value(v) for v in value]
        else:
            scalar_params[key] = format_value(value)

    combinations = list(itertools.product(*loop_values)) if loop_keys else [()]

    invocations: list[list[str]] = []
    combo_labels: list[dict[str, str]] = []

    for combo in combinations:
        cmd: list[str] = []

        if benchmark.command_prefix:
            cmd.extend(benchmark.command_prefix.split())

        cmd.append(executable)

        for key, value in scalar_params.items():
            cmd.extend([f"--{key}", value])

        for key, value in zip(loop_keys, combo):
            cmd.extend([f"--{key}", value])

        for key, values in list_of_params.items():
            cmd.append(f"--{key}")
            cmd.extend(values)

        cmd.extend(["--output-mode", "append"])

        if benchmark.log_file:
            cmd.extend(["--log-file", benchmark.log_file])

        invocations.append(cmd)
        combo_labels.append(dict(zip(loop_keys, combo)))

    return loop_keys, invocations, combo_labels


# ── Filtering ───────────────────────────────────────────────────────────


def apply_filters(
    names: set[str],
    select: list[str],
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> set[str]:
    """Apply selection and include/exclude glob patterns to benchmark names."""
    result = names

    if select:
        unknown = set(select) - names
        if unknown:
            raise SystemExit(f"Unknown benchmarks: {', '.join(sorted(unknown))}")
        result = result & set(select)

    if include_patterns:
        matched: set[str] = set()
        for pattern in include_patterns:
            matched.update(n for n in result if fnmatch.fnmatch(n, pattern))
        result = matched

    for pattern in exclude_patterns:
        result = {n for n in result if not fnmatch.fnmatch(n, pattern)}

    return result


# ── Execution ───────────────────────────────────────────────────────────


"""Execute a single benchmark invocation, print progress, and return counts."""


def _run_single_invocation(
    cmd: list[str],
    labels: dict[str, str],
    merged_env: dict[str, str] | None,
    *,
    name: str,
    i: int,
    total: int,
    show_output: bool,
    continue_on_error: bool,
) -> tuple[int, int]:
    parts = [f"{k}={v}" for k, v in labels.items()]
    progress = ", ".join(parts)
    prefix = f"[{name}] {i}/{total}"
    if progress:
        prefix += f" ({progress})"
    print(f"{prefix} ...", end="", flush=True)

    captured_stdout = ""
    captured_stderr = ""

    try:
        if show_output:
            rc = subprocess.run(cmd, env=merged_env, check=False).returncode
        else:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=merged_env,
                check=False,
            )
            rc = proc.returncode
            captured_stdout = proc.stdout
            captured_stderr = proc.stderr
    except KeyboardInterrupt:
        print(" interrupted")
        print(f"[{name}] Aborted by user.", file=sys.stderr)
        sys.exit(130)

    if rc == 0:
        print(" ok")
        return 1, 0
    fail = 1
    print(f" FAILED (exit {rc})")
    print(f"  command: {' '.join(cmd)}", file=sys.stderr)
    if captured_stdout:
        sys.stdout.write(captured_stdout)
    if captured_stderr:
        sys.stderr.write(captured_stderr)
    if not continue_on_error:
        print(f"[{name}] Aborting.", file=sys.stderr)
        sys.exit(1)
    return 0, fail


def run_benchmarks(config: Config, args: argparse.Namespace) -> None:
    # Phase 1: lightweight name-only discovery
    discovered = discover_benchmarks(
        config.common.bin_dir, config.common.benchmark_prefix
    )

    if not discovered:
        raise SystemExit("No benchmarks found.")

    # Phase 2: apply filters before expensive introspection
    selected = apply_filters(
        set(discovered.keys()),
        args.benchmark,
        args.include,
        args.exclude,
    )

    if not selected:
        raise SystemExit("No benchmarks selected after filtering.")

    # Phase 3: introspect and resolve only the selected benchmarks
    resolved = resolve_benchmarks(config, selected_names=selected)

    if not resolved:
        raise SystemExit("No benchmarks found after resolution.")

    benchmarks_to_run = {name: resolved[name] for name in sorted(selected)}

    overall_ok = 0
    overall_fail = 0
    show_output = args.verbose

    for name, benchmark in benchmarks_to_run.items():
        exe_name = (
            benchmark.final_executable or f"{config.common.benchmark_prefix}{name}"
        )
        exe_path = config.common.bin_dir / exe_name

        if not exe_path.is_file():
            print(
                f"[{name}] Executable not found: {exe_path}, skipping",
                file=sys.stderr,
            )
            continue

        if benchmark.skip_reason:
            print(f"[{name}] Skipping: {benchmark.skip_reason}", file=sys.stderr)
            continue

        merged_env: dict[str, str] | None = None
        benchmark_env = config.common.env | benchmark.env
        if benchmark_env:
            merged_env = dict(os.environ)
            merged_env.update(benchmark_env)

        _loop_keys, invocations, combo_labels = build_invocations(
            name, benchmark, config.common
        )
        total = len(invocations)

        verb = "Would run" if args.dry_run else "Running"
        print(f"[{name}] {verb} {total} invocation(s)")

        if merged_env and args.dry_run:
            overrides = {
                k: v
                for k, v in merged_env.items()
                if k not in os.environ or os.environ[k] != v
            }
            if overrides:
                env_str = " ".join(f"{k}={v}" for k, v in sorted(overrides.items()))
                print(f"  env: {env_str}")

        if (
            not args.dry_run
            and benchmark.log_file
            and config.common.output_mode == "overwrite"
        ):
            log_path = Path(benchmark.log_file)
            if log_path.exists():
                log_path.unlink()
                print(f"[{name}] Deleted existing log file: {log_path}")

        ok = 0
        fail = 0

        for i, (cmd, labels) in enumerate(zip(invocations, combo_labels), 1):
            if args.dry_run:
                print(f"  {i}. {' '.join(cmd)}")
                ok += 1
                continue

            inc_ok, inc_fail = _run_single_invocation(
                cmd,
                labels,
                merged_env,
                name=name,
                i=i,
                total=total,
                show_output=show_output,
                continue_on_error=args.continue_on_error,
            )
            ok += inc_ok
            fail += inc_fail

        if args.dry_run:
            print()
        else:
            if fail == 0:
                print(f"[{name}] Done: {ok}/{total} succeeded")
            else:
                print(f"[{name}] Done: {ok}/{total} succeeded, {fail} failed")

        overall_ok += ok
        overall_fail += fail

    if not args.dry_run and len(benchmarks_to_run) > 1:
        print(f"\nSummary: {overall_ok} succeeded, {overall_fail} failed")


# ── CLI ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TNL benchmarks from a YAML or TOML configuration file.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Path to YAML or TOML config file",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        nargs="+",
        default=[],
        metavar="NAME",
        help="Run only specified benchmark(s) by exact name",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=[],
        metavar="PATTERN",
        help="Only run benchmarks whose name matches a glob pattern (repeatable)",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        metavar="PATTERN",
        help="Skip benchmarks whose name matches a glob pattern (repeatable)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after a benchmark invocation fails",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered benchmarks and exit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show benchmark output in real time",
    )

    args = parser.parse_args()

    if not args.config.exists():
        raise SystemExit(f"Config file not found: {args.config}")

    config = load_config(args.config)

    if args.list:
        resolved = resolve_benchmarks(config)
        if not resolved:
            print("No benchmarks found.")
            return
        configured_names = set(config.benchmarks.keys())
        for name, bench in resolved.items():
            exe = bench.final_executable or f"{config.common.benchmark_prefix}{name}"
            parts = [f"  {name:30s} {exe}"]
            if name in configured_names:
                parts.append(" [configured]")
            if bench.required_params:
                params = ", ".join(f"--{p}" for p in bench.required_params)
                parts.append(f" (requires: {params})")
            if bench.list_of_params:
                params = ", ".join(f"--{p}" for p in sorted(bench.list_of_params))
                parts.append(f" (list-of: {params})")
            if bench.skip_reason:
                parts.append(f" [SKIP: {bench.skip_reason}]")
            print("".join(parts))
        return

    run_benchmarks(config, args)


if __name__ == "__main__":
    main()
