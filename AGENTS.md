# Agent Instructions

This repo is a Kubetorch workload smoke test for pinned nanoGPT training runs.

Before changing run commands or launching jobs, pull the pinned Kubetorch docs:

```bash
uv run python scripts/pull_kubetorch_docs.py
```

Read `docs/vendor/kubetorch/python_client/kubetorch/docs/guides/batch_runs.rst` before using `kt run` or `kt runs`.

Use `uv` for dependency management. Keep upstream nanoGPT pinned to `vendor/nanogpt` at commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`, and keep Kubetorch pinned to commit `47f8b30a07a3d816bdb63805d9828a7ed5bcb39c` unless deliberately updating the smoke test.
