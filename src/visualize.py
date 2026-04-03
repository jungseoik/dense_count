"""공통 시각화 유틸리티 — density map에서 heat overlay + dot overlay 생성."""
from typing import Tuple

import cv2
import numpy as np


def make_overlays(
    density_map: np.ndarray,
    original_image: np.ndarray,
    alpha: float = 0.5,
    dot_radius: int = 4,
    dot_color: Tuple[int, int, int] = (0, 255, 0),
) -> Tuple[np.ndarray, np.ndarray]:
    """density map + 원본 이미지 → (heat_overlay, dot_overlay).

    Args:
        density_map: (ph, pw) float32 density map (reduction 축소 크기).
        original_image: (H, W, 3) uint8 RGB 원본 이미지.
        alpha: 히트맵 투명도 (0~1).
        dot_radius: 점 반지름.
        dot_color: 점 색상 (BGR).

    Returns:
        heat_overlay: (H, W, 3) uint8 RGB 히트맵 오버레이.
        dot_overlay: (H, W, 3) uint8 RGB 점 오버레이.
    """
    h, w = original_image.shape[:2]
    total = float(density_map.sum())

    # 원본 해상도로 리사이즈
    d_resized = cv2.resize(density_map, (w, h), interpolation=cv2.INTER_CUBIC)

    # 히트맵 오버레이
    d_min, d_max = d_resized.min(), d_resized.max()
    ptp = d_max - d_min
    heat_u8 = np.uint8(255 * (d_resized - d_min) / max(ptp, 1e-6))
    heat_color_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    orig_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    heat_overlay_bgr = cv2.addWeighted(orig_bgr, 1 - alpha, heat_color_bgr, alpha, 0)
    heat_overlay = cv2.cvtColor(heat_overlay_bgr, cv2.COLOR_BGR2RGB)

    # 점 오버레이
    dot_overlay_bgr = orig_bgr.copy()
    dil = cv2.dilate(d_resized, np.ones((3, 3), np.float32))
    ys, xs = np.where(d_resized >= dil)
    coords = list(zip(ys, xs))
    n_dots = int(round(total))

    if len(coords) > n_dots and n_dots > 0:
        values = [d_resized[y, x] for y, x in coords]
        idxs = np.argsort(values)[::-1][:n_dots]
        chosen = [coords[i] for i in idxs]
    else:
        chosen = coords[:n_dots] if n_dots > 0 else []

    for y, x in chosen:
        cv2.circle(dot_overlay_bgr, (x, y), radius=dot_radius, color=dot_color, thickness=-1)
    dot_overlay = cv2.cvtColor(dot_overlay_bgr, cv2.COLOR_BGR2RGB)

    return heat_overlay, dot_overlay
