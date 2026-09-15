"""Shared version gates for the pinned LeRobot 0.4.4 worker stack."""

from __future__ import annotations

import re


_VERSION = re.compile(
    r"^(?P<major>[0-9]+)\.(?P<minor>[0-9]+)"
    r"(?:\.(?P<patch>[0-9]+))?(?P<suffix>[a-zA-Z0-9.+_-]*)$"
)


def release_tuple(value: object) -> tuple[int, int, int] | None:
    """Return the numeric release prefix, accepting NVIDIA/local build suffixes."""

    if not isinstance(value, str):
        return None
    match = _VERSION.fullmatch(value.strip())
    if match is None:
        return None
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch") or 0),
    )


def validate_vla_framework_versions(
    torch_version: object, torchvision_version: object
) -> None:
    """Enforce the Torch ranges declared by LeRobot 0.4.4.

    NVIDIA alpha and local build suffixes are accepted. Torch 2.11 and
    TorchVision 0.26 pre-releases are deliberately excluded because this
    repository has not revalidated the SmolVLA API with those release lines.
    """

    torch_release = release_tuple(torch_version)
    vision_release = release_tuple(torchvision_version)
    torch_suffix = _VERSION.fullmatch(str(torch_version).strip())
    vision_suffix = _VERSION.fullmatch(str(torchvision_version).strip())
    torch_at_lower_prerelease = (
        torch_release == (2, 2, 1)
        and torch_suffix is not None
        and re.match(r"(?:\.dev|a|b|rc)", torch_suffix.group("suffix")) is not None
    )
    vision_at_lower_prerelease = (
        vision_release == (0, 21, 0)
        and vision_suffix is not None
        and re.match(r"(?:\.dev|a|b|rc)", vision_suffix.group("suffix")) is not None
    )
    if (
        torch_release is None
        or torch_release < (2, 2, 1)
        or torch_at_lower_prerelease
        or torch_release[:2] >= (2, 11)
    ):
        raise ValueError(
            f"torch {torch_version!r} is outside LeRobot 0.4.4's supported range"
        )
    if (
        vision_release is None
        or vision_release < (0, 21, 0)
        or vision_at_lower_prerelease
        or vision_release[:2] >= (0, 26)
    ):
        raise ValueError(
            "torchvision "
            f"{torchvision_version!r} is outside LeRobot 0.4.4's supported range"
        )
