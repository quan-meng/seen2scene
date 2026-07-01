from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image


def save_gif(
    frames: List[Union[np.ndarray, torch.Tensor]],
    out_path: Union[str, Path],
    fps: float = 2.0,
    loop: int = 0,
) -> None:
    """Save a list of image frames as an animated GIF.

    Args:
        frames: List of frames. Each frame can be:
            - ``np.ndarray`` of shape ``[H, W, 3]`` with dtype ``uint8`` or
              ``float32`` in ``[0, 1]``.
            - ``torch.Tensor`` of shape ``[3, H, W]`` or ``[H, W, 3]`` in
              ``[0, 1]`` or ``uint8``.
        out_path: Output file path (should end in ``.gif``).
        fps: Frames per second (e.g. ``2.0`` → 500 ms per frame).
        loop: Number of times to loop (``0`` = infinite).
    """
    if not frames:
        raise ValueError("frames list is empty — nothing to save")

    pil_frames: List[Image.Image] = []
    for frame in frames:
        if isinstance(frame, torch.Tensor):
            arr = frame.cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = arr.transpose(1, 2, 0)  # [3, H, W] → [H, W, 3]
        else:
            arr = np.asarray(frame)

        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)

        pil_frames.append(Image.fromarray(arr))

    duration_ms = int(1000 / fps)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=False,
    )
