"""GPU VRAM 한계 및 배치별 추론 속도 벤치마크.

목적:
  - GPU별로 VRAM 한계치가 어떻게 달라지는지 측정
  - 배치(이미지 개수) 증가에 따른 추론 속도 변화 측정
  - SD / HD / FHD 해상도별 VRAM 사용량 프로파일링

사용법:
  python -m src.tests.test_vram_benchmark
  python -m src.tests.test_vram_benchmark --models onnx trt_mixed
  python -m src.tests.test_vram_benchmark --max-images 32
"""
import argparse
import gc
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
except Exception:
    _NVML = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import InferenceConfig
from src.onnx import ClipEBCOnnx
from src.trt import ClipEBCTensorRT

# ── 상수 ──
RESOLUTIONS = {
    "SD  (640x360)":   (360, 640),
    "HD  (1280x720)":  (720, 1280),
    "FHD (1920x1080)": (1080, 1920),
}

DEFAULT_BATCH_COUNTS = [1, 2, 4, 8, 16, 32]

MODEL_PATHS = {
    "onnx":      "assets/CLIP_EBC_nwpu_rmse_onnx.onnx",
    "trt_fp32":  "assets/CLIP_EBC_nwpu_rmse_tensorrt.engine",
    "trt_mixed": "assets/CLIP_EBC_nwpu_rmse_tensorrt_mixed.engine",
}


# ── 유틸리티 ──
def gpu_info() -> Dict[str, float]:
    if not _NVML:
        return {"used_mb": -1, "free_mb": -1, "total_mb": -1}
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return {
        "used_mb": info.used / 1024**2,
        "free_mb": info.free / 1024**2,
        "total_mb": info.total / 1024**2,
    }


def gpu_name() -> str:
    if not _NVML:
        return "Unknown"
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    return pynvml.nvmlDeviceGetName(handle)


def make_mock_images(n: int, h: int, w: int) -> List[np.ndarray]:
    return [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def tiles_for_resolution(h: int, w: int, ws: int = 224) -> int:
    rows = max(1, int(np.ceil((h - ws) / ws)) + 1)
    cols = max(1, int(np.ceil((w - ws) / ws)) + 1)
    return rows * cols


def load_model(name: str):
    path = MODEL_PATHS.get(name)
    if not path or not os.path.exists(path):
        return None
    cfg = InferenceConfig(model_path=path)
    if name == "onnx":
        return ClipEBCOnnx(cfg)
    else:
        return ClipEBCTensorRT(cfg)


def clean_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ── 테스트 1: 해상도별 VRAM 프로파일 ──
def test_vram_by_resolution(model_names: List[str]):
    print("\n" + "=" * 80)
    print(f"  VRAM 프로파일 — GPU: {gpu_name()}")
    print(f"  총 VRAM: {gpu_info()['total_mb']:.0f} MB")
    print("=" * 80)

    for mname in model_names:
        clean_gpu()
        model = load_model(mname)
        if model is None:
            print(f"\n  [{mname}] SKIP — 엔진 파일 없음")
            continue

        torch.cuda.synchronize()
        base_gpu = gpu_info()["used_mb"]

        print(f"\n  [{mname}] 모델 로드 후 GPU: {base_gpu:.0f} MB")
        print(f"  {'해상도':18s} {'tiles':>6s} {'이미지수':>8s} {'peak(MB)':>10s} {'delta(MB)':>10s} {'추론(ms)':>10s}")
        print(f"  {'-'*18} {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

        for res_label, (h, w) in RESOLUTIONS.items():
            tiles = tiles_for_resolution(h, w)
            images = make_mock_images(1, h, w)

            # warmup
            model.predict(images)
            clean_gpu()
            before = gpu_info()["used_mb"]

            torch.cuda.synchronize()
            t = time.perf_counter()
            model.predict(images)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t) * 1000

            peak = gpu_info()["used_mb"]
            delta = peak - base_gpu

            print(f"  {res_label:18s} {tiles:>6d} {1:>8d} {peak:>9.0f}MB {delta:>+9.0f}MB {elapsed:>9.1f}ms")

        del model
        clean_gpu()


# ── 테스트 2: 배치(이미지 개수) 증가별 속도 + VRAM ──
def test_batch_scaling(model_names: List[str], max_images: int = 32):
    batch_counts = [b for b in DEFAULT_BATCH_COUNTS if b <= max_images]
    if max_images not in batch_counts:
        batch_counts.append(max_images)

    print("\n" + "=" * 80)
    print(f"  배치(이미지 개수) 스케일링 — FHD (1920x1080)")
    print("=" * 80)

    h, w = 1080, 1920
    tiles_per_img = tiles_for_resolution(h, w)

    for mname in model_names:
        clean_gpu()
        model = load_model(mname)
        if model is None:
            continue

        torch.cuda.synchronize()
        base_gpu = gpu_info()["used_mb"]

        print(f"\n  [{mname}] (tiles/img={tiles_per_img})")
        print(f"  {'images':>7s} {'총tiles':>8s} {'peak(MB)':>10s} {'delta(MB)':>10s} {'총(ms)':>10s} {'img당(ms)':>10s} {'FPS':>6s}")
        print(f"  {'-'*7} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")

        for n in batch_counts:
            images = make_mock_images(n, h, w)

            try:
                model.predict(images)  # warmup
            except Exception:
                print(f"  {n:>7d}  OOM")
                break

            clean_gpu()
            torch.cuda.synchronize()
            t = time.perf_counter()
            model.predict(images)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t) * 1000

            peak = gpu_info()["used_mb"]
            delta = peak - base_gpu
            per_img = elapsed / n
            fps = 1000.0 / per_img if per_img > 0 else 0

            total_tiles = tiles_per_img * n
            print(f"  {n:>7d} {total_tiles:>8d} {peak:>9.0f}MB {delta:>+9.0f}MB {elapsed:>9.1f}ms {per_img:>9.1f}ms {fps:>5.1f}")

            del images
            clean_gpu()

        del model
        clean_gpu()


# ── 테스트 3: VRAM 한계치 탐색 ──
def test_vram_limit(model_names: List[str]):
    print("\n" + "=" * 80)
    print(f"  VRAM 한계치 탐색 — FHD (1920x1080)")
    print(f"  목적: GPU별 최대 동시 처리 가능 이미지 수 확인")
    print("=" * 80)

    h, w = 1080, 1920

    for mname in model_names:
        clean_gpu()
        model = load_model(mname)
        if model is None:
            continue

        torch.cuda.synchronize()
        total_vram = gpu_info()["total_mb"]
        base_gpu = gpu_info()["used_mb"]
        free_vram = total_vram - base_gpu

        print(f"\n  [{mname}] 여유 VRAM: {free_vram:.0f} MB")

        last_ok = 0
        for n in [1, 2, 4, 8, 16, 32, 64, 128]:
            images = make_mock_images(n, h, w)
            try:
                clean_gpu()
                torch.cuda.synchronize()
                model.predict(images)
                torch.cuda.synchronize()
                peak = gpu_info()["used_mb"]
                delta = peak - base_gpu
                print(f"    {n:>4d} images: peak={peak:.0f}MB  delta=+{delta:.0f}MB  OK")
                last_ok = n
            except Exception as e:
                print(f"    {n:>4d} images: OOM")
                break
            finally:
                del images
                clean_gpu()

        print(f"  -> [{mname}] FHD 최대 동시 처리: {last_ok}장")

        del model
        clean_gpu()


# ── 메인 ──
def main():
    parser = argparse.ArgumentParser(description="GPU VRAM 벤치마크")
    parser.add_argument("--models", nargs="+", default=["onnx", "trt_fp32", "trt_mixed"],
                        choices=["onnx", "trt_fp32", "trt_mixed"])
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--skip-limit", action="store_true", help="VRAM 한계 탐색 건너뛰기")
    args = parser.parse_args()

    available = [m for m in args.models if os.path.exists(MODEL_PATHS.get(m, ""))]
    if not available:
        print("사용 가능한 모델이 없습니다. 엔진을 먼저 빌드하세요.")
        return

    print(f"\nGPU: {gpu_name()}")
    print(f"VRAM: {gpu_info()['total_mb']:.0f} MB")
    print(f"테스트 모델: {available}")

    test_vram_by_resolution(available)
    test_batch_scaling(available, max_images=args.max_images)

    if not args.skip_limit:
        test_vram_limit(available)

    print("\n" + "=" * 80)
    print("  벤치마크 완료")
    print("=" * 80)


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
    import assets
    main()
