"""Native (numpy / imageio) video preprocessing for Nemotron-3-Nano-Omni.

Mirrors the source `video_processing.py` + `video_io.py` + `evs.py` chain
without torch / torchvision / decord dependencies.

Pipeline:
  1. Frame extraction: decode video via `imageio[ffmpeg]`, return (T, H, W, 3) uint8.
  2. Frame sampling: uniform N frames (default 32) or fps-based.
  3. Resize each frame to (image_size, image_size) bicubic (max_num_tiles=1).
  4. CLIP-style normalize ((x - mean) / std).
  5. Pad N to a multiple of `video_temporal_patch_dim` (T=2 for this model)
     by repeating the last frame.
  6. Stack T frames into channel dim: (N//T, T*3, H, W). The source then
     feeds this to RADIO with `video_embedder` swapped in for `embedder`.

EVS (Efficient Video Sampling):
  Computes per-token retention mask based on cosine similarity between
  consecutive frames' patch tokens. Drops `q` (default 0.7 = 70%) of the
  most-similar tokens. Applied AFTER the vision tower runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# CLIP normalization (matches source `norm_mean`/`norm_std`)
NORM_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
NORM_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


# ── Frame extraction ───────────────────────────────────────────────────────

def decode_video(
    path: str | Path,
    *,
    target_frames: int = 32,
    target_fps: Optional[float] = None,
) -> Tuple[np.ndarray, dict]:
    """Decode video file → (n_frames, H, W, 3) uint8 numpy array + metadata.

    Backend priority: `imageio[ffmpeg]` (preferred, pure-pip) → `av` (PyAV).
    Both produce identical output (RGB uint8 frames).

    Args:
        path: video file path.
        target_frames: if `target_fps` is None, sample exactly this many
            frames uniformly. Default 32.
        target_fps: if set, sample at this fps regardless of `target_frames`.

    Returns:
        (frames, metadata) where metadata has `fps`, `total_frames`, `duration_s`.
    """
    path = str(path)
    try:
        import imageio.v3 as iio
        meta = iio.immeta(path, plugin="FFMPEG")
        fps = float(meta.get("fps", 30.0))
        # Probe duration to compute total frames
        try:
            duration_s = float(meta.get("duration", 0.0))
            total_frames = int(round(fps * duration_s))
        except (TypeError, ValueError, OverflowError):
            duration_s = 0.0
            total_frames = 0
        # Decode all frames first (small videos < 1 min are cheap)
        all_frames = []
        for frame in iio.imiter(path, plugin="FFMPEG"):
            all_frames.append(frame)
        frames_np = np.stack(all_frames, axis=0).astype(np.uint8)
        actual_total = frames_np.shape[0]
        if total_frames == 0:
            total_frames = actual_total
            duration_s = total_frames / max(fps, 1e-6)
    except (ImportError, Exception) as e:
        # Fallback: PyAV
        try:
            import av  # noqa: F401
        except ImportError:
            raise ImportError(
                "Video decoding needs `imageio[ffmpeg]` or `av` (PyAV). "
                "Install with `pip install imageio[ffmpeg]`."
            ) from e
        import av
        container = av.open(path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        all_frames = []
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="rgb24")
            all_frames.append(arr)
        container.close()
        frames_np = np.stack(all_frames, axis=0).astype(np.uint8)
        total_frames = frames_np.shape[0]
        duration_s = total_frames / max(fps, 1e-6)

    # Sample frames
    if target_fps is not None and target_fps > 0:
        desired = max(1, int(round(duration_s * target_fps)))
    else:
        desired = max(1, target_frames)
    if desired >= frames_np.shape[0]:
        sampled = frames_np
    elif desired == 1:
        sampled = frames_np[:1]
    else:
        idxs = np.unique(np.round(np.linspace(0, frames_np.shape[0] - 1, desired)).astype(int))
        sampled = frames_np[idxs]

    metadata = {
        "fps": fps,
        "total_frames": int(total_frames),
        "duration_s": float(duration_s),
        "n_sampled": int(sampled.shape[0]),
    }
    return sampled, metadata


# ── Resize + normalize ─────────────────────────────────────────────────────

def _bicubic_resize_pil(frames_uint8: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize each frame in (T, H, W, 3) uint8 to (T, target_h, target_w, 3)
    via PIL.Image.BICUBIC (matches source video_processing's torchvision
    InterpolationMode.BICUBIC closely enough for inference)."""
    from PIL import Image
    out = np.empty((frames_uint8.shape[0], target_h, target_w, 3), dtype=np.uint8)
    for i, frame in enumerate(frames_uint8):
        img = Image.fromarray(frame).resize((target_w, target_h), Image.BICUBIC)
        out[i] = np.asarray(img)
    return out


def preprocess_video(
    path: str | Path,
    *,
    image_size: int = 512,
    target_frames: int = 32,
    video_temporal_patch_dim: int = 2,
    target_fps: Optional[float] = None,
) -> Tuple[np.ndarray, dict]:
    """Full video preprocessing → pixel_values_videos ready for RADIO.

    Returns:
        pixel_values: (N_temporal_groups, T*3, H, W) float32 — stacked frames
            ready for the RADIO `video_embedder` linear projector (which
            expects 3*T*P*P inputs per patch).
        metadata: dict with `n_frames`, `n_temporal_groups`, `fps`, etc.

    Note: the model's `extract_video_feature` does the channel-stacking
    swap internally; for our native MLX path we emit the stacked layout
    directly so we can run RADIO with `video=True`.
    """
    frames, meta = decode_video(path, target_frames=target_frames, target_fps=target_fps)
    n_frames = frames.shape[0]

    # Pad to multiple of T by repeating the last frame
    T = video_temporal_patch_dim
    if n_frames % T != 0:
        pad_count = T - (n_frames % T)
        last = frames[-1:]
        pad = np.repeat(last, pad_count, axis=0)
        frames = np.concatenate([frames, pad], axis=0)
        n_frames = frames.shape[0]

    # Resize each frame to image_size × image_size
    frames = _bicubic_resize_pil(frames, image_size, image_size)

    # Float32 + CLIP normalize: (T, H, W, 3) → (T, 3, H, W) and apply
    arr = frames.astype(np.float32) / 255.0
    arr = arr.transpose(0, 3, 1, 2)
    mean = NORM_MEAN.reshape(1, 3, 1, 1)
    std = NORM_STD.reshape(1, 3, 1, 1)
    arr = (arr - mean) / std

    # Stack T temporal frames into channel dim:
    #   (N, 3, H, W) → (N/T, T*3, H, W)
    n_groups = n_frames // T
    arr = arr.reshape(n_groups, T * 3, image_size, image_size)

    metadata = {
        **meta,
        "n_frames_after_pad": n_frames,
        "n_temporal_groups": n_groups,
        "image_size": image_size,
    }
    return arr.astype(np.float32), metadata


# ── EVS (Efficient Video Sampling) ─────────────────────────────────────────

def compute_evs_retention_mask(
    video_embeds: np.ndarray,
    *,
    n_temporal_groups: int,
    grid_h: int,
    grid_w: int,
    spatial_merge_size: int = 2,
    q: float = 0.7,
) -> np.ndarray:
    """Drop `q` fraction of patch tokens with highest similarity to previous frame.

    Mirrors `evs.py::EfficientVideoSampling.compute_retention_mask`.

    Args:
        video_embeds: (T*Hm*Wm, hidden) where T=n_temporal_groups,
            Hm=grid_h//spatial_merge_size, Wm=grid_w//spatial_merge_size.
        n_temporal_groups: T (number of temporal patches).
        grid_h, grid_w: full patch grid (before spatial-merge).
        spatial_merge_size: 2 for our pixel_shuffle 0.5 path (shuffle factor=2).
        q: pruning rate. 0.7 means drop 70% of tokens (keep 30%).

    Returns:
        retention_mask: (T*Hm*Wm,) boolean — True = keep.
    """
    Tg = n_temporal_groups
    Hm = grid_h // spatial_merge_size
    Wm = grid_w // spatial_merge_size
    expected = Tg * Hm * Wm
    if video_embeds.shape[0] != expected:
        raise ValueError(
            f"EVS expected {expected} tokens "
            f"(T={Tg} × Hm={Hm} × Wm={Wm}) but got {video_embeds.shape[0]}"
        )

    # Reshape to (T, Hm, Wm, C)
    embeds = video_embeds.reshape(Tg, Hm, Wm, -1).astype(np.float32)

    # Cosine similarity between frame[t] and frame[t-1], per spatial patch.
    a = embeds[1:, ...]
    b = embeds[:-1, ...]
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    similarity = (a_norm * b_norm).sum(axis=-1)        # (T-1, Hm, Wm)
    dissimilarity = 1.0 - similarity

    # First frame: mark dissimilarity=255 so we always keep all its tokens.
    first = np.full((1, Hm, Wm), 255.0, dtype=np.float32)
    dissimilarity = np.concatenate([first, dissimilarity], axis=0)  # (T, Hm, Wm)
    dis_flat = dissimilarity.reshape(-1)

    min_num_tokens = Hm * Wm                            # one full frame
    evs_num_tokens = int(Tg * min_num_tokens * (1.0 - q))
    n_keep = max(min_num_tokens, evs_num_tokens)

    # Top-k highest dissimilarity (= least redundant)
    order = np.argsort(-dis_flat, kind="stable")
    keep_idx = order[:n_keep]
    mask = np.zeros_like(dis_flat, dtype=bool)
    mask[keep_idx] = True
    return mask
