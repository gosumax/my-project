"""Start native Codex with a separately selected MAIN model and reasoning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-model", help="Override only the MAIN model")
    parser.add_argument("--main-reasoning", help="Override only the MAIN reasoning level")
    parser.add_argument("--check-only", action="store_true", help="Print the command without starting a session")
    parser.add_argument("--prompt", help="Optional first task message")
    args = parser.parse_args(argv)

    try:
        with (PROJECT_ROOT / ".codex" / "config.toml").open("rb") as source:
            config = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        parser.error(f"Cannot read project Codex config: {exc}")

    model = args.main_model if args.main_model is not None else config.get("model")
    reasoning = args.main_reasoning if args.main_reasoning is not None else config.get("model_reasoning_effort")
    for name, value in (("main-model", model), ("main-reasoning", reasoning)):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", value):
            parser.error(f"{name} must be a non-empty model/level identifier")

    executable = shutil.which("codex")
    if executable is None:
        parser.error("codex executable was not found in PATH")

    # Explicitly select both MAIN fields so user/global defaults cannot supply
    # the other half. Never override agents.*, permissions, profiles or prompts.
    command = [
        executable, "-C", str(PROJECT_ROOT), "-m", model,
        "-c", f"model_reasoning_effort={json.dumps(reasoning)}",
    ]
    if args.prompt is not None:
        command.extend(["--", args.prompt])
    print(f"MAIN MODEL: {model}\nMAIN REASONING: {reasoning}", flush=True)
    if args.check_only:
        print(json.dumps({"cwd": str(PROJECT_ROOT), "argv": command}, ensure_ascii=False))
        print("Command check only; model availability and runtime settings are not verified.")
        return 0
    try:
        return subprocess.call(command, cwd=PROJECT_ROOT)
    except OSError as exc:
        parser.error(f"Cannot start Codex: {exc}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
