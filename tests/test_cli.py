import json
from typer.testing import CliRunner

from nanogpt_kubetorch import cli


runner = CliRunner()


def test_env_probe_writes_environment(tmp_path):
    result = runner.invoke(cli.app, ["env-probe", "--output-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    env = json.loads((tmp_path / "environment.json").read_text())
    assert env["kubetorch_commit"] == "47f8b30a07a3d816bdb63805d9828a7ed5bcb39c"
    assert env["nanogpt_commit"] == "3adf61e154c3fe3fca428ad6bc3818b27a3b8291"


def test_introspect_run_invokes_kt_commands(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, cwd=None):
        calls.append(command)
        if command[:3] == ["kt", "runs", "show"]:
            return '{"run_id":"run-1","status":"succeeded","notes":[],"artifacts":[]}'
        return "logs here"

    monkeypatch.setattr(cli, "run_command", fake_run)

    result = runner.invoke(
        cli.app,
        ["introspect-run", "run-1", "--output-dir", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert calls == [["kt", "runs", "show", "run-1"], ["kt", "runs", "logs", "run-1"]]
    summary = json.loads((tmp_path / "run-1" / "introspection_summary.json").read_text())
    assert summary["run_id"] == "run-1"
    assert summary["log_bytes"] == len("logs here")


def test_module_execution_runs_cli(tmp_path):
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nanogpt_kubetorch.cli",
            "env-probe",
            "--output-dir",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "environment.json").exists()
