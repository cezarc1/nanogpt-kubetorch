from __future__ import annotations

import json
import subprocess
from pathlib import Path

import typer

from .constants import KUBETORCH_COMMIT, NANOGPT_COMMIT
from .reporting import safe_note, write_json
from .training import environment_manifest, run_training

app = typer.Typer(add_completion=False)


def run_command(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)
    return result.stdout


@app.command("env-probe")
def env_probe(output_dir: Path = typer.Option(Path("results"), "--output-dir")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = environment_manifest("env-probe")
    manifest["kubetorch_commit"] = KUBETORCH_COMMIT
    manifest["nanogpt_commit"] = NANOGPT_COMMIT
    write_json(output_dir / "environment.json", manifest)
    safe_note(f"nanoGPT env probe recorded at {output_dir / 'environment.json'}")


@app.command("smoke")
def smoke(
    output_dir: Path = typer.Option(Path("results"), "--output-dir"),
    namespace: str = typer.Option("kubetorch", "--namespace"),
    device: str = typer.Option("cuda", "--device"),
) -> None:
    run_training("smoke", output_dir=output_dir, namespace=namespace, device=device)


@app.command("baseline")
def baseline(
    output_dir: Path = typer.Option(Path("results"), "--output-dir"),
    namespace: str = typer.Option("kubetorch", "--namespace"),
    device: str = typer.Option("cuda", "--device"),
) -> None:
    run_training("baseline", output_dir=output_dir, namespace=namespace, device=device)


@app.command("introspect-run")
def introspect_run(
    run_id: str = typer.Argument(...),
    output_dir: Path = typer.Option(Path("introspection"), "--output-dir"),
) -> None:
    target = output_dir / run_id
    target.mkdir(parents=True, exist_ok=True)
    show_raw = run_command(["kt", "runs", "show", run_id])
    logs_raw = run_command(["kt", "runs", "logs", run_id])
    (target / "run.json").write_text(show_raw)
    (target / "logs.txt").write_text(logs_raw)
    try:
        run = json.loads(show_raw)
    except json.JSONDecodeError:
        run = {"run_id": run_id, "parse_error": True}
    summary = {
        "run_id": run_id,
        "status": run.get("status"),
        "source_key": run.get("source_key"),
        "logs_key": run.get("logs_key"),
        "image": run.get("image"),
        "resources": run.get("resources"),
        "notes_count": len(run.get("notes") or []),
        "artifacts_count": len(run.get("artifacts") or []),
        "log_bytes": len(logs_raw),
        "commands": [["kt", "runs", "show", run_id], ["kt", "runs", "logs", run_id]],
    }
    write_json(target / "introspection_summary.json", summary)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
