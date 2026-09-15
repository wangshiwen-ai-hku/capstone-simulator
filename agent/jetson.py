"""Jetson release profiles used by hardware acceptance reports.

The marketing JetPack version is derived from Jetson Linux (L4T), which is
the on-device source of truth.  Package metadata is useful diagnostics but is
not required because a minimal installation may omit the nvidia-jetpack meta
package.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class JetPackProfile:
    version: str
    l4t_release: int
    l4t_revision: str
    cuda_runtime_version: int


# Profiles supported by the AGX Orin hardware branch.  NVIDIA's public archive
# has no JetPack 7.1.2 release; 7.1 is L4T 38.4 and does not support Orin.
JETPACK_PROFILES = {
    "7.2": JetPackProfile("7.2", 39, "2", 13020),
    "7.2.1": JetPackProfile("7.2.1", 39, "2.1", 13020),
}

_L4T = re.compile(
    r"\bR(?P<release>[0-9]+)\b.*?\bREVISION:\s*(?P<revision>[0-9]+(?:\.[0-9]+)*)"
)


def normalize_jetpack_profile(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("JetPack profile must be a version string")
    normalized = value.strip().removeprefix("JetPack").strip().removeprefix("v")
    if normalized not in JETPACK_PROFILES:
        supported = ", ".join(JETPACK_PROFILES)
        if normalized == "7.1.2":
            raise ValueError(
                "JetPack 7.1.2 is not an NVIDIA release; AGX Orin JetPack 7 "
                "support starts at 7.2. Verify /etc/nv_tegra_release (7.2.1 is "
                "R39 revision 2.1) before testing"
            )
        raise ValueError(
            f"unsupported JetPack profile {value!r}; supported profiles: {supported}"
        )
    return normalized


def parse_jetson_linux(value: object) -> tuple[int, str] | None:
    if not isinstance(value, str):
        return None
    match = _L4T.search(value.replace("\n", " "))
    if match is None:
        return None
    return int(match.group("release")), match.group("revision")


def jetpack_profile_evidence(
    profile: str,
    *,
    jetson_linux: object,
    cuda_runtime_version: object | None = None,
) -> dict:
    """Validate exact L4T evidence and CUDA Runtime when the caller supplies it."""

    normalized = normalize_jetpack_profile(profile)
    assert normalized is not None
    expected = JETPACK_PROFILES[normalized]
    observed_l4t = parse_jetson_linux(jetson_linux)
    l4t_matches = observed_l4t == (expected.l4t_release, expected.l4t_revision)
    runtime_supplied = cuda_runtime_version is not None
    runtime_matches = (
        type(cuda_runtime_version) is int
        and cuda_runtime_version == expected.cuda_runtime_version
        if runtime_supplied
        else None
    )
    return {
        "profile": normalized,
        "profile_scope": "l4t_and_cuda_runtime" if runtime_supplied else "l4t_only",
        "passed": l4t_matches and (runtime_matches is not False),
        "jetson_linux_matches": l4t_matches,
        "cuda_runtime_matches": runtime_matches,
        "expected": {
            "l4t_release": expected.l4t_release,
            "l4t_revision": expected.l4t_revision,
            "cuda_runtime_version": expected.cuda_runtime_version,
        },
        "observed": {
            "jetson_linux": jetson_linux,
            "parsed_l4t": list(observed_l4t) if observed_l4t else None,
            "cuda_runtime_version": cuda_runtime_version,
        },
    }


def resolve_required_jetpack(
    required: str | None, *, legacy_require_jetpack721: bool = False
) -> str | None:
    normalized = normalize_jetpack_profile(required)
    if legacy_require_jetpack721:
        if normalized not in (None, "7.2.1"):
            raise ValueError(
                "--require-jetpack721 conflicts with the selected JetPack profile"
            )
        return "7.2.1"
    return normalized
