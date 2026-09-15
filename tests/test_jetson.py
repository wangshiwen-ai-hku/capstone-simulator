import pytest

from agent.jetson import (
    jetpack_profile_evidence,
    normalize_jetpack_profile,
    parse_jetson_linux,
    resolve_required_jetpack,
)


def test_l4t_is_the_exact_jetpack_profile_source() -> None:
    release = "# R39 (release), REVISION: 2.1, GCID: 123"
    assert parse_jetson_linux(release) == (39, "2.1")
    evidence = jetpack_profile_evidence(
        "7.2.1", jetson_linux=release, cuda_runtime_version=13020
    )
    assert evidence["passed"] is True
    assert evidence["profile_scope"] == "l4t_and_cuda_runtime"
    assert evidence["observed"]["parsed_l4t"] == [39, "2.1"]


def test_l4t_only_profile_is_explicit_when_runtime_is_not_supplied() -> None:
    evidence = jetpack_profile_evidence(
        "7.2.1", jetson_linux="# R39 (release), REVISION: 2.1,"
    )
    assert evidence["passed"] is True
    assert evidence["profile_scope"] == "l4t_only"
    assert evidence["cuda_runtime_matches"] is None


@pytest.mark.parametrize(
    "release,runtime",
    [
        ("# R39 (release), REVISION: 2.10,", 13020),
        ("# R39 (release), REVISION: 2.1,", 13010),
        ("# R38 (release), REVISION: 4.0,", 13020),
        (None, 13020),
    ],
)
def test_profile_mismatch_fails_closed(release, runtime) -> None:
    assert (
        jetpack_profile_evidence(
            "7.2.1", jetson_linux=release, cuda_runtime_version=runtime
        )["passed"]
        is False
    )


def test_nonexistent_jetpack_712_is_rejected_with_actionable_diagnostic() -> None:
    with pytest.raises(ValueError, match=r"7\.1\.2 is not an NVIDIA release"):
        normalize_jetpack_profile("7.1.2")


def test_jetpack_profile_must_be_a_string() -> None:
    with pytest.raises(ValueError, match="version string"):
        normalize_jetpack_profile(True)  # type: ignore[arg-type]


def test_legacy_721_gate_resolves_to_the_named_profile() -> None:
    assert resolve_required_jetpack(None, legacy_require_jetpack721=True) == "7.2.1"
    with pytest.raises(ValueError, match="conflicts"):
        resolve_required_jetpack("7.2", legacy_require_jetpack721=True)
