from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import KUBETORCH_COMMIT, NANOGPT_COMMIT, NANOGPT_DIR, NANOGPT_REPO
from .reporting import IssueLedger, publish_file, safe_note, write_json

EVAL_RE = re.compile(r"step (?P<step>\d+): train loss (?P<train>[0-9.]+), val loss (?P<val>[0-9.]+)")
ITER_RE = re.compile(
    r"iter (?P<iter>\d+): loss (?P<loss>[0-9.]+), time (?P<time>[0-9.]+)ms, mfu (?P<mfu>-?[0-9.]+)%"
)


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: Path
    nanogpt_dir: Path = NANOGPT_DIR
    device: str = "cuda"
    namespace: str = "kubetorch"


def build_smoke_command(config: TrainingConfig) -> list[str]:
    out_dir = config.output_dir / "out-shakespeare-char-smoke"
    return [
        "python",
        "train.py",
        "config/train_shakespeare_char.py",
        f"--out_dir={out_dir}",
        "--max_iters=20",
        "--lr_decay_iters=20",
        "--eval_interval=10",
        "--eval_iters=10",
        "--log_interval=1",
        "--batch_size=16",
        "--block_size=128",
        "--n_layer=4",
        "--n_head=4",
        "--n_embd=128",
        "--compile=False",
        f"--device={config.device}",
    ]


def build_baseline_command(config: TrainingConfig) -> list[str]:
    out_dir = config.output_dir / "out-shakespeare-char"
    return [
        "python",
        "train.py",
        "config/train_shakespeare_char.py",
        f"--out_dir={out_dir}",
        f"--device={config.device}",
        "--compile=False",
    ]


def parse_iter_metrics(text: str) -> dict[str, list[dict[str, float | int]]]:
    evals = []
    iters = []
    for line in text.splitlines():
        if match := EVAL_RE.search(line):
            evals.append(
                {
                    "step": int(match.group("step")),
                    "train_loss": float(match.group("train")),
                    "val_loss": float(match.group("val")),
                }
            )
        if match := ITER_RE.search(line):
            iters.append(
                {
                    "iter": int(match.group("iter")),
                    "loss": float(match.group("loss")),
                    "time_ms": float(match.group("time")),
                    "mfu_percent": float(match.group("mfu")),
                }
            )
    return {"evals": evals, "iters": iters}


def best_val_loss(metrics: dict[str, list[dict[str, Any]]]) -> float | None:
    vals = [float(row["val_loss"]) for row in metrics.get("evals", [])]
    return min(vals) if vals else None


def run_command(command: list[str], *, cwd: Path, log_path: Path | None = None) -> str:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output: list[str] = []
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a")
    else:
        handle = None
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            output.append(line)
            if handle:
                handle.write(line)
                handle.flush()
    finally:
        if handle:
            handle.close()
    code = proc.wait()
    text = "".join(output)
    if code != 0:
        raise RuntimeError(f"command failed with exit code {code}: {' '.join(command)}\n{text[-2000:]}")
    return text


def environment_manifest(command: str) -> dict[str, Any]:
    manifest = {
        "command": command,
        "python": sys.version,
        "platform": platform.platform(),
        "kubetorch_commit": KUBETORCH_COMMIT,
        "nanogpt_repo": NANOGPT_REPO,
        "nanogpt_commit": NANOGPT_COMMIT,
        "kt_run_id": os.getenv("KT_RUN_ID"),
        "kt_namespace": os.getenv("KT_NAMESPACE"),
        "kt_workdir_key": os.getenv("KT_WORKDIR_KEY"),
        "kt_logs_key": os.getenv("KT_LOGS_KEY"),
    }
    try:
        import torch

        manifest["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        manifest["torch_error"] = str(exc)
    return manifest


def prepare_shakespeare(config: TrainingConfig, train_log: Path) -> None:
    run_command(["python", "data/shakespeare_char/prepare.py"], cwd=config.nanogpt_dir, log_path=train_log)


def sample_model(config: TrainingConfig, out_dir: Path, sample_path: Path) -> None:
    command = [
        "python",
        "sample.py",
        f"--out_dir={out_dir}",
        f"--device={config.device}",
        "--compile=False",
        "--num_samples=2",
        "--max_new_tokens=200",
    ]
    sample = run_command(command, cwd=config.nanogpt_dir)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(sample)


def publish_outputs(namespace: str, output_dir: Path) -> None:
    for path in sorted(output_dir.glob("*")):
        if path.is_file():
            publish_file(namespace=namespace, path=path, name=path.name)


def run_training(kind: str, *, output_dir: Path, namespace: str = "kubetorch", device: str = "cuda") -> dict[str, Any]:
    output_dir = output_dir.resolve()
    config = TrainingConfig(output_dir=output_dir, namespace=namespace, device=device)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = IssueLedger(output_dir / "issues.md")
    train_log = output_dir / "train.log"
    command_builder = build_smoke_command if kind == "smoke" else build_baseline_command
    command = command_builder(config)
    started = time.monotonic()

    write_json(output_dir / "environment.json", environment_manifest(kind))
    write_json(
        output_dir / "run_config.json",
        {
            "kind": kind,
            "command": command,
            "namespace": namespace,
            "device": device,
            "nanogpt_commit": NANOGPT_COMMIT,
            "kubetorch_commit": KUBETORCH_COMMIT,
        },
    )
    safe_note(f"nanoGPT {kind} run starting; output_dir={output_dir}")

    try:
        prepare_shakespeare(config, train_log)
        train_output = run_command(command, cwd=config.nanogpt_dir, log_path=train_log)
        metrics = parse_iter_metrics(train_log.read_text() + train_output)
        best = best_val_loss(metrics)
        out_dir = output_dir / ("out-shakespeare-char-smoke" if kind == "smoke" else "out-shakespeare-char")
        sample_model(config, out_dir, output_dir / "sample.txt")
        elapsed = time.monotonic() - started
        write_json(output_dir / "metrics.json", {"best_val_loss": best, **metrics})
        write_json(output_dir / "performance.json", {"elapsed_seconds": elapsed, "kind": kind})
        write_json(
            output_dir / "dataset_manifest.json",
            {
                "dataset": "shakespeare_char",
                "train_bin": str(config.nanogpt_dir / "data/shakespeare_char/train.bin"),
                "val_bin": str(config.nanogpt_dir / "data/shakespeare_char/val.bin"),
            },
        )
        safe_note(f"nanoGPT {kind} run succeeded; best_val_loss={best}")
    except Exception as exc:
        ledger.exception(category="training", summary=f"nanoGPT {kind} failed", exc=exc)
        safe_note(f"nanoGPT {kind} run failed: {exc}")
        raise
    finally:
        publish_outputs(namespace, output_dir)

    return {"kind": kind, "best_val_loss": best, "output_dir": str(output_dir)}
