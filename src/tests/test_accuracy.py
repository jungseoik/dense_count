"""ONNX / TRT FP32 / TRT Mixed 정확도 비교 테스트.

목적:
  - 3개 백엔드 간 count 차이, density map 상관계수 비교
  - GPU 교체 시 TRT 엔진 정확도 변화 확인
  - 해상도별 정확도 안정성 검증

사용법:
  python -m src.tests.test_accuracy
  python -m src.tests.test_accuracy --images assets/289.jpg assets/sample.png
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import InferenceConfig
from src.onnx import ClipEBCOnnx
from src.trt import ClipEBCTensorRT

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
except Exception:
    _NVML = False

MODEL_PATHS = {
    "onnx":      "assets/CLIP_EBC_nwpu_rmse_onnx.onnx",
    "trt_fp32":  "assets/CLIP_EBC_nwpu_rmse_tensorrt.engine",
    "trt_mixed": "assets/CLIP_EBC_nwpu_rmse_tensorrt_mixed.engine",
}

DEFAULT_IMAGES = ["assets/289.jpg", "assets/sample.png", "assets/sample2.png"]


def gpu_name() -> str:
    if not _NVML:
        return "Unknown"
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    return pynvml.nvmlDeviceGetName(handle)


def load_model(name: str):
    path = MODEL_PATHS.get(name)
    if not path or not os.path.exists(path):
        return None
    cfg = InferenceConfig(model_path=path)
    if name == "onnx":
        return ClipEBCOnnx(cfg)
    return ClipEBCTensorRT(cfg)


def compute_density_map(model, image_np: np.ndarray) -> np.ndarray:
    """predict 호출 후 내부 density map을 추출하기 위해 단일 이미지로 재조립."""
    from torchvision.transforms import Normalize, ToTensor
    from PIL import Image as PILImage

    cfg = model.config
    to_tensor = ToTensor()
    normalize = Normalize(mean=cfg.mean, std=cfg.std)

    pil = PILImage.fromarray(image_np)
    processed = normalize(to_tensor(pil)).unsqueeze(0).numpy()
    h, w = processed.shape[-2:]
    ws, st = cfg.window_size, cfg.stride
    reduction = cfg.reduction

    num_rows = max(1, int(np.ceil((h - ws) / st)) + 1)
    num_cols = max(1, int(np.ceil((w - ws) / st)) + 1)

    wins, poses = [], []
    for r in range(num_rows):
        for c in range(num_cols):
            x0 = min(r * st, h - ws) if h > ws else 0
            y0 = min(c * st, w - ws) if w > ws else 0
            wins.append(processed[:, :, x0:x0 + ws, y0:y0 + ws])
            poses.append((x0, y0))

    batch = np.vstack(wins)

    if hasattr(model, "session"):
        preds = model.session.run([model.output_name], {model.input_name: batch})[0]
    else:
        preds = model._run_inference(batch)

    ph, pw = h // reduction, w // reduction
    pred_map = np.zeros((1, ph, pw), dtype=np.float32)
    count_map = np.zeros_like(pred_map)

    for i, (x0, y0) in enumerate(poses):
        pred = preds[i]
        xo, yo = x0 // reduction, y0 // reduction
        pred_h, pred_w = pred.shape[1:]
        x1 = min(xo + pred_h, ph)
        y1 = min(yo + pred_w, pw)
        pred_map[:, xo:x1, yo:y1] += pred[:, :x1 - xo, :y1 - yo]
        count_map[:, xo:x1, yo:y1] += 1.0

    count_map[count_map == 0] = 1.0
    pred_map /= count_map
    return pred_map.squeeze()


def test_accuracy(image_paths: list):
    print("\n" + "=" * 80)
    print(f"  정확도 비교 — GPU: {gpu_name()}")
    print("=" * 80)

    models = {}
    for name in ["onnx", "trt_fp32", "trt_mixed"]:
        m = load_model(name)
        if m is not None:
            models[name] = m

    if "onnx" not in models:
        print("  ONNX 모델 없음. 중단.")
        return

    images = []
    names = []
    for p in image_paths:
        if os.path.exists(p):
            images.append(np.array(Image.open(p).convert("RGB")))
            names.append(os.path.basename(p))

    if not images:
        print("  테스트 이미지 없음.")
        return

    # ONNX 기준값
    onnx = models["onnx"]
    onnx_counts = onnx.predict(images)

    print(f"\n  기준 (ONNX):")
    for i, name in enumerate(names):
        h, w = images[i].shape[:2]
        print(f"    {name:20s} ({w}x{h}) -> count={onnx_counts[i]:.4f}")

    # 비교
    comparisons = [(k, v) for k, v in models.items() if k != "onnx"]

    for model_name, model in comparisons:
        print(f"\n  [{model_name}] vs ONNX:")
        print(f"  {'이미지':20s} {'ONNX':>10s} {model_name:>12s} {'diff':>8s} {'corr':>10s} {'MAE':>12s} {'판정':>6s}")
        print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*8} {'-'*10} {'-'*12} {'-'*6}")

        all_pass = True
        for i, name in enumerate(names):
            trt_counts = model.predict([images[i]])
            tc = trt_counts[0]
            diff = abs(onnx_counts[i] - tc)

            # density map 비교
            onnx_dm = compute_density_map(onnx, images[i])
            trt_dm = compute_density_map(model, images[i])

            if onnx_dm.shape == trt_dm.shape:
                flat_o = onnx_dm.flatten()
                flat_t = trt_dm.flatten()
                corr = np.corrcoef(flat_o, flat_t)[0, 1] if np.std(flat_o) > 0 else 1.0
                mae = np.abs(onnx_dm - trt_dm).mean()
            else:
                corr = 0.0
                mae = float("inf")

            threshold = 2.0 if "mixed" in model_name else 1.0
            ok = diff < threshold and corr > 0.95
            if not ok:
                all_pass = False

            status = "PASS" if ok else "FAIL"
            print(f"  {name:20s} {onnx_counts[i]:>10.2f} {tc:>12.2f} {diff:>8.4f} {corr:>10.6f} {mae:>12.8f} {status:>6s}")

        overall = "PASS" if all_pass else "FAIL"
        print(f"\n  [{model_name}] 종합: [{overall}]")


def test_speed(image_paths: list):
    print("\n" + "=" * 80)
    print("  추론 속도 비교")
    print("=" * 80)

    images = [np.array(Image.open(p).convert("RGB")) for p in image_paths if os.path.exists(p)]
    if not images:
        return

    N = 10
    WARMUP = 3

    models = {}
    for name in ["onnx", "trt_fp32", "trt_mixed"]:
        m = load_model(name)
        if m is not None:
            models[name] = m

    print(f"\n  입력: {len(images)}개 이미지, {N}회 반복")
    print(f"  {'모델':12s} {'평균(ms)':>10s} {'최소(ms)':>10s} {'img당(ms)':>10s} {'FPS':>8s}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    import torch

    for name, model in models.items():
        for _ in range(WARMUP):
            model.predict(images)

        times = []
        for _ in range(N):
            torch.cuda.synchronize()
            t = time.perf_counter()
            model.predict(images)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1000)

        avg = np.mean(times)
        mn = np.min(times)
        per_img = avg / len(images)
        fps = 1000.0 / per_img
        print(f"  {name:12s} {avg:>9.1f}ms {mn:>9.1f}ms {per_img:>9.1f}ms {fps:>7.1f}")


def main():
    parser = argparse.ArgumentParser(description="정확도 비교 테스트")
    parser.add_argument("--images", nargs="+", default=DEFAULT_IMAGES)
    args = parser.parse_args()

    test_accuracy(args.images)
    test_speed(args.images)

    print("\n" + "=" * 80)
    print("  정확도 테스트 완료")
    print("=" * 80)


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
    import assets
    main()
