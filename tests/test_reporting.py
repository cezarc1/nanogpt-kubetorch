import json

from nanogpt_kubetorch.reporting import kt_uri, write_jsonl, write_json


def test_write_json_and_jsonl_create_parent_dirs(tmp_path):
    manifest = tmp_path / "nested" / "manifest.json"
    rows = tmp_path / "nested" / "rows.jsonl"

    write_json(manifest, {"run": "smoke"})
    write_jsonl(rows, [{"i": 1}, {"i": 2}])

    assert json.loads(manifest.read_text()) == {"run": "smoke"}
    assert rows.read_text().splitlines() == ['{"i": 1}', '{"i": 2}']


def test_kt_uri_normalizes_slashes():
    assert kt_uri("kubetorch", "/experiments/nanogpt/run/metrics.json") == (
        "kt://kubetorch/experiments/nanogpt/run/metrics.json"
    )
