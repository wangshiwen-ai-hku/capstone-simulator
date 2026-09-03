"""Asset integrity and dependency isolation checks; no GPU execution is simulated."""

import copy
import json
from pathlib import Path
import subprocess
import venv

import pytest

from examples.vla_workloads.bundle import (
    POLICY_ID,
    POLICY_REVISION,
    VLM_ID,
    VLM_REVISION,
    file_sha256,
    validate_bundle,
)
from scripts.install_vla import torch_constraints


def _bundle(root: Path) -> dict:
    names = (
        "policy/config.json",
        "policy/model.safetensors",
        "policy/policy_preprocessor.json",
        "policy/policy_postprocessor.json",
        "policy/policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy/policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "vlm/config.json",
        "vlm/tokenizer.json",
        "vlm/tokenizer_config.json",
    )
    files = {}
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("asset-integrity-test fixture; not model weights")
        files[name] = file_sha256(path)
    manifest = {
        "schema": "mars.vla.model-bundle.v1",
        "policy": {"repo_id": POLICY_ID, "revision": POLICY_REVISION},
        "vlm": {"repo_id": VLM_ID, "revision": VLM_REVISION},
        "files": files,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_bundle_detects_changed_download_before_model_load(tmp_path):
    manifest = _bundle(tmp_path)
    assert validate_bundle(tmp_path) == manifest
    (tmp_path / "policy/model.safetensors").write_text("truncated download")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_bundle(tmp_path)


def test_bundle_rejects_unverified_extra_weight_file(tmp_path):
    _bundle(tmp_path)
    (tmp_path / "policy/pytorch_model.bin").write_bytes(b"unrecorded")
    with pytest.raises(ValueError, match="unrecorded"):
        validate_bundle(tmp_path)


@pytest.mark.parametrize(
    "mutation", ("missing_normalization", "wrong_revision", "outside_path")
)
def test_bundle_rejects_incomplete_or_different_assets(tmp_path, mutation):
    manifest = copy.deepcopy(_bundle(tmp_path))
    if mutation == "missing_normalization":
        del manifest["files"][
            "policy/policy_preprocessor_step_5_normalizer_processor.safetensors"
        ]
    elif mutation == "wrong_revision":
        manifest["policy"]["revision"] = "main"
    else:
        manifest["files"]["policy/../../outside"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        validate_bundle(tmp_path)


def test_dependency_constraints_preserve_nvidia_local_builds():
    assert torch_constraints("2.8.0+nv25.08", "0.23.0+abc123") == (
        "torch===2.8.0+nv25.08\ntorchvision===0.23.0+abc123\n"
    )
    assert "2.7.0a0+7c8ec84dab" in torch_constraints(
        "2.7.0a0+7c8ec84dab", "0.22.0a0+abc"
    )


@pytest.mark.parametrize(
    "torch_version,vision_version",
    (("2.11.0", "0.23.0"), ("2.8.0", "0.26.0"), ("2.8.0\nother-package", "0.23.0")),
)
def test_incompatible_dependency_constraints_fail(torch_version, vision_version):
    with pytest.raises(ValueError):
        torch_constraints(torch_version, vision_version)


def test_worker_interpreter_preserves_virtual_environment(tmp_path):
    from agent.executor import VlaExecutor

    environment = tmp_path / "ml-environment"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    executor = VlaExecutor("io", worker_python=str(environment / "bin/python"))
    prefix = subprocess.check_output(
        [
            executor.worker_python,
            "-c",
            "import sys; print(sys.prefix)",
        ],
        text=True,
    ).strip()
    assert Path(prefix) == environment
