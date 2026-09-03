"""Pinned, offline model bundles. This module has no MARS or ML dependencies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re


POLICY_ID = "lerobot/smolvla_base"
POLICY_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
VLM_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
DATASET_ID = "lerobot/svla_so100_pickplace"
DATASET_REVISION = "728583b5eaf9e739a7f119e2def466fa1d552402"
LEROBOT_VERSION = "0.4.4"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_bundle(directory: str | Path) -> dict:
    """Check every required file before allowing the local model to run."""
    root = Path(directory).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size > 100_000:
        raise ValueError(
            "missing/oversized model manifest; run scripts.prepare_vla model"
        )
    manifest = json.loads(manifest_path.read_text())
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "mars.vla.model-bundle.v1"
    ):
        raise ValueError("unsupported VLA model bundle schema")
    for key, repo, revision in (
        ("policy", POLICY_ID, POLICY_REVISION),
        ("vlm", VLM_ID, VLM_REVISION),
    ):
        if manifest.get(key) != {"repo_id": repo, "revision": revision}:
            raise ValueError(
                f"unsupported {key} checkpoint; this runner uses a pinned SmolVLA bundle"
            )
    files = manifest.get("files")
    required = {
        "policy/config.json",
        "policy/model.safetensors",
        "policy/policy_preprocessor.json",
        "policy/policy_postprocessor.json",
        "policy/policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy/policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "vlm/config.json",
        "vlm/tokenizer.json",
        "vlm/tokenizer_config.json",
    }
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError(
            "model bundle is missing weights, tokenizer or normalization statistics"
        )
    for name, expected in files.items():
        if not isinstance(name, str) or not name:
            raise ValueError("invalid model manifest file name")
        relative = PurePosixPath(name)
        if (
            not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[0] not in {"policy", "vlm"}
            or not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
        ):
            raise ValueError("invalid model manifest file entry")
        path = root.joinpath(*relative.parts).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"model file missing or outside bundle: {name}")
        if file_sha256(path) != expected:
            raise ValueError(f"model file checksum mismatch: {name}")
    # Loading must not discover unverified weight/config files beside verified ones.
    actual = {
        str(p.relative_to(root))
        for folder in ("policy", "vlm")
        for p in (root / folder).rglob("*")
        if p.is_file() and ".cache" not in p.relative_to(root).parts
    }
    if actual != set(files):
        raise ValueError("model bundle contains unrecorded or missing files")
    return manifest


def download_bundle(directory: str | Path) -> dict:
    from huggingface_hub import snapshot_download

    root = Path(directory).resolve()
    if (root / "manifest.json").exists():
        return validate_bundle(root)
    root.mkdir(parents=True, exist_ok=True)
    for folder, repo, revision in (
        ("policy", POLICY_ID, POLICY_REVISION),
        ("vlm", VLM_ID, VLM_REVISION),
    ):
        snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=root / folder,
            # The strict policy checkpoint contains the VLM parameters too.
            # Only the backbone config/tokenizer is needed for construction.
            allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"]
            + (["*.safetensors"] if folder == "policy" else []),
            ignore_patterns=["onnx/*"],
        )
    files = {
        str(path.relative_to(root)): file_sha256(path)
        for folder in ("policy", "vlm")
        for path in sorted((root / folder).rglob("*"))
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    }
    manifest = {
        "schema": "mars.vla.model-bundle.v1",
        "policy": {"repo_id": POLICY_ID, "revision": POLICY_REVISION},
        "vlm": {"repo_id": VLM_ID, "revision": VLM_REVISION},
        "files": files,
    }
    with (root / "manifest.json").open("x") as output:
        json.dump(manifest, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    return validate_bundle(root)
