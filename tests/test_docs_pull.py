import json

from scripts.pull_kubetorch_docs import build_manifest


def test_build_manifest_records_pinned_docs():
    manifest = build_manifest(["python_client/kubetorch/docs/guides/batch_runs.rst"])

    assert manifest["repo"] == "https://github.com/cezarc1/kubetorch"
    assert manifest["commit"] == "47f8b30a07a3d816bdb63805d9828a7ed5bcb39c"
    assert manifest["fetched_paths"] == ["python_client/kubetorch/docs/guides/batch_runs.rst"]
    json.dumps(manifest)
