from typing import List, Tuple, Union

import os

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision.transforms import Normalize, ToTensor

from ..config import InferenceConfig
from ..visualize import make_overlays


class ClipEBCOnnx:
    """ONNX 기반 CLIP-EBC 군중 계수 모델."""

    def __init__(self, config: InferenceConfig):
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(
                f"ONNX 모델 파일을 찾을 수 없습니다: {config.model_path}\n"
                f"  'import assets'로 자동 다운로드하거나, 경로를 확인하세요."
            )

        self.config = config
        self.window_size = config.window_size
        self.stride = config.stride
        self.reduction = config.reduction
        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=config.mean, std=config.std)

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(
            config.model_path, sess_options=session_options, providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _run_inference(self, batch: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: batch})[0]

    def _predict_core(self, images: List[np.ndarray]) -> Tuple[List[float], List[np.ndarray]]:
        """추론 + density map 재조립.

        Returns:
            counts: 각 이미지의 예측 군중 수.
            density_maps: 각 이미지의 density map (ph, pw).
        """
        all_windows = []
        stitch_info = []

        for image_np in images:
            pil_image = Image.fromarray(image_np)
            normalized = self.normalize(self.to_tensor(pil_image)).unsqueeze(0).numpy()
            h, w = normalized.shape[-2:]
            ws, st = self.window_size, self.stride

            num_rows = max(1, int(np.ceil((h - ws) / st)) + 1)
            num_cols = max(1, int(np.ceil((w - ws) / st)) + 1)

            wins, poses = [], []
            for r in range(num_rows):
                for c in range(num_cols):
                    x0 = min(r * st, h - ws) if h > ws else 0
                    y0 = min(c * st, w - ws) if w > ws else 0
                    wins.append(normalized[:, :, x0:x0 + ws, y0:y0 + ws])
                    poses.append((x0, y0))

            all_windows.extend(wins)
            stitch_info.append({"shape": (h, w), "num": len(wins), "pos": poses})

        if not all_windows:
            return [0.0] * len(images), [np.zeros((1, 1)) for _ in images]

        batch = np.vstack(all_windows)
        preds = self._run_inference(batch)

        counts = []
        density_maps = []
        cursor = 0
        for info in stitch_info:
            num = info["num"]
            h, w = info["shape"]
            ph, pw = h // self.reduction, w // self.reduction

            dmap = np.zeros((ph, pw), dtype=np.float32)
            cmap = np.zeros_like(dmap)

            for i, (x0, y0) in enumerate(info["pos"]):
                pred = preds[cursor + i][0]
                xo, yo = x0 // self.reduction, y0 // self.reduction
                hh, ww = pred.shape
                x1, y1 = min(xo + hh, ph), min(yo + ww, pw)
                dmap[xo:x1, yo:y1] += pred[:x1 - xo, :y1 - yo]
                cmap[xo:x1, yo:y1] += 1

            cursor += num
            cmap[cmap == 0] = 1
            dmap /= cmap
            counts.append(float(dmap.sum()))
            density_maps.append(dmap)

        return counts, density_maps

    def predict(self, images: List[np.ndarray]) -> List[float]:
        """배치 추론. Returns: 각 이미지의 예측 군중 수."""
        counts, _ = self._predict_core(images)
        return counts

    def predict_single(self, image: Union[str, np.ndarray]) -> float:
        """단일 이미지 추론."""
        if isinstance(image, str):
            image = np.array(Image.open(image).convert("RGB"))
        return self.predict([image])[0]

    def predict_dense_dot(
        self, images: List[np.ndarray], alpha: float = 0.5,
    ) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
        """추론 + 시각화.

        Returns:
            counts, heat_overlays, dot_overlays
        """
        counts, density_maps = self._predict_core(images)
        heat_overlays = []
        dot_overlays = []

        for dmap, orig in zip(density_maps, images):
            heat, dot = make_overlays(dmap, orig, alpha=alpha)
            heat_overlays.append(heat)
            dot_overlays.append(dot)

        return counts, heat_overlays, dot_overlays
