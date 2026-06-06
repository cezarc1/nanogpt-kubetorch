from pathlib import Path


def test_runtime_image_installs_compiler_for_torch_compile():
    dockerfile = Path("Dockerfile").read_text()

    assert "build-essential" in dockerfile
