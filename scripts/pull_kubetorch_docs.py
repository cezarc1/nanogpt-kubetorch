from __future__ import annotations

import json
from pathlib import Path

import httpx

KUBETORCH_REPO = "https://github.com/cezarc1/kubetorch"
KUBETORCH_RAW = "https://raw.githubusercontent.com/cezarc1/kubetorch"
KUBETORCH_COMMIT = "47f8b30a07a3d816bdb63805d9828a7ed5bcb39c"
DOC_PATHS = [
    "python_client/kubetorch/docs/guides/batch_runs.rst",
    "python_client/kubetorch/docs/index.rst",
]


def build_manifest(paths: list[str]) -> dict[str, object]:
    return {
        "repo": KUBETORCH_REPO,
        "commit": KUBETORCH_COMMIT,
        "fetched_paths": paths,
    }


def fetch_docs(output_dir: Path = Path("docs/vendor/kubetorch")) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fetched = []
    for path in DOC_PATHS:
        url = f"{KUBETORCH_RAW}/{KUBETORCH_COMMIT}/{path}"
        response = httpx.get(url, timeout=30.0)
        response.raise_for_status()
        dest = output_dir / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(response.text)
        fetched.append(path)
    manifest = build_manifest(fetched)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    fetch_docs()
