import json
import os
import sys
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Normalize, ToTensor

from ..config import InferenceConfig
from ..visualize import make_overlays

# models/ 패키지 임포트를 위해 프로젝트 루트를 path에 추가
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class ClipEBCPth:
    """PyTorch 기반 CLIP-EBC 군중 계수 모델.

    models.get_model()로 PyTorch 모델을 로드하여 추론합니다.
    """

    def __init__(
        self,
        config: InferenceConfig,
        model_name: str = "clip_vit_b_16",
        truncation: int = 4,
        granularity: str = "fine",
        anchor_points_type: str = "average",
        dataset_name: str = "qnrf",
        num_vpt: int = 32,
        vpt_drop: float = 0.0,
        deep_vpt: bool = True,
        prompt_type: str = "word",
        config_dir: str = "configs",
    ):
        self.config = config
        self.window_size = config.window_size
        self.stride = config.stride
        self.reduction = config.reduction
        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=config.mean, std=config.std)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 설정 로드
        config_path = os.path.join(config_dir, f"reduction_{config.reduction}.json")
        with open(config_path, "r") as f:
            model_config = json.load(f)[str(truncation)][dataset_name]

        bins = [(float(b[0]), float(b[1])) for b in model_config["bins"][granularity]]
        anchor_points = [float(p) for p in model_config["anchor_points"][granularity][anchor_points_type]]

        if not os.path.exists(config.model_path):
            raise FileNotFoundError(
                f"PyTorch 체크포인트를 찾을 수 없습니다: {config.model_path}\n"
                f"  'import assets'로 자동 다운로드하거나, 경로를 확인하세요."
            )

        # PyTorch 모델 로드
        from models import get_model

        self.model = get_model(
            backbone=model_name,
            input_size=config.window_size,
            reduction=config.reduction,
            bins=bins,
            anchor_points=anchor_points,
            prompt_type=prompt_type,
            num_vpt=num_vpt,
            vpt_drop=vpt_drop,
            deep_vpt=deep_vpt,
        )
        ckpt = torch.load(config.model_path, map_location=self.device)
        self.model.load_state_dict(ckpt)
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _run_inference(self, batch: np.ndarray) -> np.ndarray:
        """PyTorch 모델로 배치 추론."""
        tensor = torch.from_numpy(batch).to(self.device)
        output = self.model(tensor)
        return output.cpu().numpy()

    def _predict_core(self, images: List[np.ndarray]) -> Tuple[List[float], List[np.ndarray]]:
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
        counts, _ = self._predict_core(images)
        return counts

    def predict_single(self, image: Union[str, np.ndarray]) -> float:
        if isinstance(image, str):
            image = np.array(Image.open(image).convert("RGB"))
        return self.predict([image])[0]

    def predict_dense_dot(
        self, images: List[np.ndarray], alpha: float = 0.5,
    ) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
        counts, density_maps = self._predict_core(images)
        heat_overlays, dot_overlays = [], []
        for dmap, orig in zip(density_maps, images):
            heat, dot = make_overlays(dmap, orig, alpha=alpha)
            heat_overlays.append(heat)
            dot_overlays.append(dot)
        return counts, heat_overlays, dot_overlays
