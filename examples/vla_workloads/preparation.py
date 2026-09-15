"""Export an actual recorded robot observation; never fabricate sensor values."""

from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path

from .bundle import DATASET_ID, DATASET_REVISION


def export_sample(output: str | Path, cache: str | Path) -> dict:
    """Download the pinned dataset episode and export its first real frame.

    LeRobot v3 stores episodes in consolidated videos. The two source videos
    total about 470 MB even though only one observation is exported.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    destination = Path(output)
    if destination.exists():
        raise ValueError("observation file exists; choose a new output name")
    dataset = LeRobotDataset(
        DATASET_ID,
        root=Path(cache).resolve() / DATASET_REVISION,
        episodes=[0],
        revision=DATASET_REVISION,
        video_backend="pyav",
    )
    frame = dataset[0]
    images = {}
    mapping = {
        "observation.images.top": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    }
    for source, target in mapping.items():
        array = frame[source].detach().cpu().permute(1, 2, 0).numpy()
        # Downsize actual pixels before transport. No synthetic missing view.
        image = Image.fromarray((array.clip(0, 1) * 255).round().astype("uint8"))
        image.thumbnail((256, 256), Image.Resampling.LANCZOS)
        stream = BytesIO()
        image.save(stream, format="PNG")
        images[target] = {
            "encoding": "png",
            "data_base64": base64.b64encode(stream.getvalue()).decode("ascii"),
        }
    sample = {
        "schema": "mars.vla.observation.v1",
        "task": frame["task"],
        "state": frame["observation.state"].detach().cpu().tolist(),
        "images": images,
        "provenance": {
            "source": "recorded_robot_dataset",
            "repo_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "episode_index": 0,
            "frame_index": 0,
            "robot_type": "so100",
            "camera_mapping": mapping,
            "missing_camera_keys": ["observation.images.camera3"],
            "resize": "preserve aspect ratio, longest edge <=256, PNG",
            "state_joint_order": [
                "main_shoulder_pan",
                "main_shoulder_lift",
                "main_elbow_flex",
                "main_wrist_flex",
                "main_wrist_roll",
                "main_gripper",
            ],
            "purpose": "pretrained inference smoke test; not a task-success evaluation",
        },
    }
    from .pipeline import validate_observation

    # Use the same schema validator as the IO Agent, without importing MARS.
    encoded = json.dumps(sample, ensure_ascii=False, allow_nan=False).encode()
    if len(encoded) > 1_500_000:
        raise ValueError("exported observation exceeds the transfer budget")
    validate_observation(sample)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(encoded + b"\n")
    return {
        "output": str(destination.resolve()),
        "source": sample["provenance"],
        "image_keys": list(images),
        "state_dim": len(sample["state"]),
        "task": sample["task"],
        "bytes": len(encoded),
    }
