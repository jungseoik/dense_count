import os
from typing import List, Tuple, Union

import numpy as np
import tensorrt as trt
import torch
from PIL import Image
from torchvision.transforms import Normalize, ToTensor

from ..config import InferenceConfig
from ..visualize import make_overlays


class ClipEBCTensorRT:
    """TensorRT 기반 CLIP-EBC 군중 계수 모델."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.window_size = config.window_size
        self.stride = config.stride
        self.reduction = config.reduction
        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=config.mean, std=config.std)

        self._load_engine(config.model_path)
        self._init_buffers()

    def _load_engine(self, engine_path: str):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(
                f"TensorRT 엔진 파일을 찾을 수 없습니다: {engine_path}\n"
                f"  빌드: python -m src.trt.builder --mixed --output {engine_path}"
            )

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_name = self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        self.engine_max_batch = self.engine.get_tensor_profile_shape(self.input_name, 0)[2][0]

    def _init_buffers(self):
        self._alloc(min(64, self.engine_max_batch))
        self.stream = torch.cuda.Stream()

    def _alloc(self, batch_size: int):
        ws = self.config.window_size
        input_shape = (batch_size, 3, ws, ws)
        self.context.set_input_shape(self.input_name, input_shape)
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        self._out_per_sample = output_shape[1:]
        self.input_buffer = torch.empty(input_shape, dtype=torch.float32, device="cuda")
        self.output_buffer = torch.empty(output_shape, dtype=torch.float32, device="cuda")
        self._buf_size = batch_size

    def _ensure_buffers(self, n: int):
        if n > self._buf_size:
            self._alloc(n)

    def _infer(self, batch: np.ndarray) -> np.ndarray:
        n = batch.shape[0]
        self._ensure_buffers(n)
        ws = self.config.window_size
        self.context.set_input_shape(self.input_name, (n, 3, ws, ws))
        self.input_buffer[:n].copy_(torch.from_numpy(batch))
        self.context.set_tensor_address(self.input_name, self.input_buffer.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output_buffer.data_ptr())
        with torch.cuda.stream(self.stream):
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        return self.output_buffer[:n].cpu().numpy()

    def _run_inference(self, batch: np.ndarray) -> np.ndarray:
        total = batch.shape[0]
        if total <= self.engine_max_batch:
            return self._infer(batch)
        chunks = []
        for i in range(0, total, self.engine_max_batch):
            chunks.append(self._infer(batch[i:i + self.engine_max_batch]))
        return np.concatenate(chunks, axis=0)

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
