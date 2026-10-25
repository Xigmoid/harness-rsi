"""Tests for the IFEval task builder (``scripts/build_ifeval_tasks.py``)."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUILDER = ROOT / "scripts" / "build_ifeval_tasks.py"
DATASET = ROOT / "tests" / "data" / "ifeval_input_data.jsonl"

pytestmark = pytest.mark.skipif(
    not DATASET.exists(), reason="the vendored IFEval dataset is missing"
)


def _run_builder(out: Path, limit: int = 3) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--out",
            str(out),
            "--data",
            str(DATASET),
            "--limit",
            str(limit),
            "--write-base",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("ifeval-tasks")
    result = _run_builder(out)
    assert result.returncode == 0, result.stderr
    return out


def test_builder_writes_tasks(built: Path) -> None:
    tasks = sorted(p for p in built.glob("prompt-*") if p.is_dir())
    assert len(tasks) == 3
    for task in tasks:
        assert (task / "task.toml").exists()
        assert (task / "instruction.md").exists()
        assert (task / "tests" / "test.sh").exists()
        assert (task / "tests" / "test_outputs.py").exists()


def test_task_toml_is_valid_toml_with_the_base_image(built: Path) -> None:
    for task in sorted(built.glob("prompt-*")):
        config = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
        assert config["schema_version"] == "1.1"
        assert config["environment"]["docker_image"] == "rsi/ifeval-base:1"
        assert config["task"]["name"].startswith("ifeval/prompt-")
        assert config["agent"]["timeout_sec"] > 0
        assert config["verifier"]["timeout_sec"] > 0


def test_base_build_context_is_written(built: Path) -> None:
    dockerfile = (built / "base" / "Dockerfile").read_text(encoding="utf-8")
    assert "pytest" in dockerfile
    assert "langdetect" in dockerfile
    assert "nltk" in dockerfile
    assert "punkt" in dockerfile
    # the rsi package has to be importable by the agent's isolated venv
    assert "PYTHONPATH=/opt/rsi" in dockerfile
    assert "COPY rsi /opt/rsi/rsi" in dockerfile
    assert (built / "base" / ".dockerignore").exists()


def test_generated_verifier_is_valid_python(built: Path) -> None:
    for task in sorted(built.glob("prompt-*")):
        source = (task / "tests" / "test_outputs.py").read_text(encoding="utf-8")
        compile(source, str(task), "exec")


def test_generated_verifier_rejects_a_bad_response(built: Path, tmp_path: Path) -> None:
    """The verifier must fail a response that follows no instruction."""
    task = sorted(built.glob("prompt-*"))[0]
    source = (task / "tests" / "test_outputs.py").read_text(encoding="utf-8")
    response_file = tmp_path / "response.txt"
    response_file.write_text("x", encoding="utf-8")
    patched = source.replace('"/app/response.txt"', repr(str(response_file)))

    namespace: dict = {}
    exec(compile(patched, "verifier", "exec"), namespace)  # noqa: S102
    with pytest.raises(AssertionError):
        namespace["test_response_follows_every_instruction"]()


def test_instruction_md_contains_the_prompt(built: Path) -> None:
    rows = {
        json.loads(line)["key"]: json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    for task in sorted(built.glob("prompt-*")):
        key = int(task.name.split("-")[1])
        prompt = rows[key]["prompt"]
        text = (task / "instruction.md").read_text(encoding="utf-8")
        assert prompt in text
        assert "/app/response.txt" in text
