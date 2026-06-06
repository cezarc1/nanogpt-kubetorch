# nanoGPT Kubetorch

Kubetorch batch-run smoke tests for a pinned nanoGPT workload.

- Upstream nanoGPT: `karpathy/nanoGPT@3adf61e154c3fe3fca428ad6bc3818b27a3b8291`
- Kubetorch fork: `cezarc1/kubetorch@47f8b30a07a3d816bdb63805d9828a7ed5bcb39c`

## Pull pinned Kubetorch docs

```bash
uv run python scripts/pull_kubetorch_docs.py
```

## Local tests

```bash
uv run pytest -q
```

## Build image

```bash
docker buildx build \
  --platform linux/amd64 \
  -t ghcr.io/cezarc1/nanogpt-kubetorch:$(git rev-parse --short=8 HEAD) \
  --push .
```

## Kubetorch runs

```bash
kt run \
  --name nanogpt-smoke \
  --intent "nanoGPT smoke: validate Kubetorch source snapshot, Job execution, CUDA, dataset prep, logs, notes, artifacts, and introspection" \
  --namespace kubetorch \
  --image ghcr.io/cezarc1/nanogpt-kubetorch:$(git rev-parse --short=8 HEAD) \
  --source-dir . \
  --resources '{"requests":{"cpu":"4","memory":"12Gi","nvidia.com/gpu":"1"},"limits":{"cpu":"8","memory":"20Gi","nvidia.com/gpu":"1"}}' \
  -- \
  python -m nanogpt_kubetorch.cli smoke --output-dir results --namespace kubetorch
```

After completion, inspect with `kt runs list`, `kt runs show RUN_ID`, and `kt runs logs RUN_ID`.
