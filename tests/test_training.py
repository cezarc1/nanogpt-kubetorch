from pathlib import Path

from nanogpt_kubetorch.training import (
    TrainingConfig,
    best_val_loss,
    build_baseline_command,
    build_smoke_command,
    parse_iter_metrics,
)


def test_build_smoke_command_uses_reduced_shakespeare_config():
    config = TrainingConfig(output_dir=Path("results"))

    command = build_smoke_command(config)

    assert command[:3] == ["python", "train.py", "config/train_shakespeare_char.py"]
    assert "--out_dir=results/out-shakespeare-char-smoke" in command
    assert "--max_iters=20" in command
    assert "--eval_interval=10" in command
    assert "--compile=False" in command
    assert "--device=cuda" in command


def test_build_baseline_command_uses_full_schedule_with_eager_execution():
    config = TrainingConfig(output_dir=Path("results"))

    command = build_baseline_command(config)

    assert command == [
        "python",
        "train.py",
        "config/train_shakespeare_char.py",
        "--out_dir=results/out-shakespeare-char",
        "--device=cuda",
        "--compile=False",
    ]


def test_parse_iter_metrics_extracts_loss_time_and_mfu():
    text = """
step 0: train loss 4.2871, val loss 4.2823
iter 10: loss 2.9142, time 43.20ms, mfu 12.40%
step 20: train loss 2.1000, val loss 1.5900
"""

    metrics = parse_iter_metrics(text)

    assert metrics["evals"][-1] == {"step": 20, "train_loss": 2.1, "val_loss": 1.59}
    assert metrics["iters"] == [{"iter": 10, "loss": 2.9142, "time_ms": 43.2, "mfu_percent": 12.4}]
    assert best_val_loss(metrics) == 1.59
