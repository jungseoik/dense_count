"""CLIP-EBC CLI 추론 도구.

사용법:
    python main.py --image assets/289.jpg
    python main.py --image assets/289.jpg --visualize all --save
    python main.py --image assets/289.jpg --backend trt_mixed
"""
import argparse
import os

import numpy as np
from PIL import Image

import assets
from src.config import InferenceConfig


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP-EBC Crowd Counting")
    parser.add_argument("--image", required=True, help="입력 이미지 경로")
    parser.add_argument("--backend", default="onnx", choices=["onnx", "trt_fp32", "trt_mixed"],
                        help="추론 백엔드 (기본: onnx)")
    parser.add_argument("--visualize", choices=["density", "dots", "all", "none"],
                        default="none", help="시각화 타입")
    parser.add_argument("--save", action="store_true", help="시각화 결과 저장")
    parser.add_argument("--output-dir", default="results", help="저장 디렉토리")
    parser.add_argument("--alpha", type=float, default=0.5, help="히트맵 투명도")
    return parser.parse_args()


MODEL_PATHS = {
    "onnx": "assets/CLIP_EBC_nwpu_rmse_onnx.onnx",
    "trt_fp32": "assets/CLIP_EBC_nwpu_rmse_tensorrt.engine",
    "trt_mixed": "assets/CLIP_EBC_nwpu_rmse_tensorrt_mixed.engine",
}


def main():
    args = parse_args()

    config = InferenceConfig(model_path=MODEL_PATHS[args.backend])

    if args.backend == "onnx":
        from src.onnx import ClipEBCOnnx
        model = ClipEBCOnnx(config)
    else:
        from src.trt import ClipEBCTensorRT
        model = ClipEBCTensorRT(config)

    img = np.array(Image.open(args.image).convert("RGB"))

    if args.visualize == "none":
        count = model.predict_single(img)
        print(f"예측된 군중 수: {count:.2f}")
    else:
        counts, heats, dots = model.predict_dense_dot([img], alpha=args.alpha)
        print(f"예측된 군중 수: {counts[0]:.2f}")

        if args.save:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.visualize in ("density", "all"):
            if args.save:
                path = os.path.join(args.output_dir, "density_map.png")
                Image.fromarray(heats[0]).save(path)
                print(f"저장: {path}")

        if args.visualize in ("dots", "all"):
            if args.save:
                path = os.path.join(args.output_dir, "dot_map.png")
                Image.fromarray(dots[0]).save(path)
                print(f"저장: {path}")


if __name__ == "__main__":
    main()
