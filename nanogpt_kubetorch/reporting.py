from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .constants import EXPERIMENT_PREFIX


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return target


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return target


def kt_uri(namespace: str, key: str) -> str:
    return f"kt://{namespace}/{key.strip('/')}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact_key(filename: str, run_id: str | None = None) -> str:
    run = run_id or os.getenv("KT_RUN_ID") or "local"
    return f"{EXPERIMENT_PREFIX}/{run}/{filename}"


def safe_note(body: str, author: str = "agent") -> bool:
    try:
        from kubetorch import runs

        runs.note(body, author=author)
        return True
    except Exception:
        return False


def safe_artifact(
    *,
    name: str,
    uri: str,
    kind: str = "kt-data-store",
    metadata: dict[str, Any] | None = None,
    author: str = "agent",
) -> bool:
    try:
        from kubetorch import runs

        runs.artifact(name=name, uri=uri, kind=kind, metadata=metadata, author=author)
        return True
    except Exception:
        return False


def publish_file(
    *,
    namespace: str,
    path: str | Path,
    name: str,
    kind: str = "kt-data-store",
    metadata: dict[str, Any] | None = None,
) -> bool:
    source = Path(path)
    key = artifact_key(source.name)
    try:
        from kubetorch.data_store import DataStoreClient

        DataStoreClient(namespace=namespace).put(key=key, src=source, force=True)
        return safe_artifact(name=name, uri=kt_uri(namespace, key), kind=kind, metadata=metadata)
    except Exception:
        return False


class IssueLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(
                "# nanoGPT Kubetorch Issues\n\n"
                "| category | severity | summary | evidence | workaround | recorded_at |\n"
                "| --- | --- | --- | --- | --- | --- |\n"
            )

    def add(
        self,
        *,
        category: str,
        severity: str,
        summary: str,
        evidence: str = "",
        workaround: str = "",
    ) -> None:
        row = [category, severity, summary, evidence, workaround, now_iso()]
        safe = [str(value).replace("\n", " ").replace("|", "\\|") for value in row]
        with self.path.open("a") as handle:
            handle.write("| " + " | ".join(safe) + " |\n")

    def exception(self, *, category: str, summary: str, exc: BaseException) -> None:
        self.add(
            category=category,
            severity="high",
            summary=summary,
            evidence="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            workaround="inspect kt runs logs and the published issue ledger",
        )
