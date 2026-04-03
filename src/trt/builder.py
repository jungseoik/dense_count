import os
import time

import tensorrt as trt

from ..config import TRTBuildConfig

FP32_LAYER_PATTERNS = ["Softmax", "ReduceL2", "/Exp", "/ReduceSum"]


def _should_force_fp32(layer_name: str) -> bool:
    return any(p in layer_name for p in FP32_LAYER_PATTERNS)


def build_engine(config: TRTBuildConfig) -> str:
    """TRTBuildConfig 기반으로 TensorRT 엔진을 빌드합니다.

    Returns:
        생성된 엔진 파일 경로.
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    if not os.path.exists(config.onnx_path):
        raise FileNotFoundError(
            f"ONNX 모델 파일을 찾을 수 없습니다: {config.onnx_path}\n"
            f"  'import assets'로 자동 다운로드하거나, 경로를 확인하세요."
        )

    print(f"ONNX 모델 파싱 중: {config.onnx_path}")
    with open(config.onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  파싱 오류: {parser.get_error(i)}")
            raise RuntimeError("ONNX 모델 파싱 실패")

    input_tensor = network.get_input(0)
    spatial = tuple(input_tensor.shape[1:])
    print(f"입력: {input_tensor.name}, spatial={spatial}")

    # 빌더 설정
    bconfig = builder.create_builder_config()
    bconfig.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, config.workspace_gb << 30)

    use_fp16 = (config.fp16 or config.mixed_precision) and builder.platform_has_fast_fp16

    if use_fp16:
        bconfig.set_flag(trt.BuilderFlag.FP16)

    if config.mixed_precision and use_fp16:
        bconfig.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        fp32_count = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if _should_force_fp32(layer.name):
                layer.precision = trt.DataType.FLOAT
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.DataType.FLOAT)
                fp32_count += 1
        print(f"Mixed Precision: FP16 기본 + {fp32_count}개 레이어 FP32 고정")
    elif use_fp16:
        print("FP16 전체 활성화")
    else:
        print("FP32 모드")

    # 최적화 프로필
    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_tensor.name,
        (1,) + spatial,
        (config.opt_batch_size,) + spatial,
        (config.max_batch_size,) + spatial,
    )
    bconfig.add_optimization_profile(profile)
    print(f"프로필: min=1, opt={config.opt_batch_size}, max={config.max_batch_size}")

    # 빌드
    print("엔진 빌드 중...")
    start = time.time()
    serialized = builder.build_serialized_network(network, bconfig)
    elapsed = time.time() - start

    if serialized is None:
        raise RuntimeError("TensorRT 엔진 빌드 실패")

    print(f"빌드 완료: {elapsed:.1f}초")

    os.makedirs(os.path.dirname(config.output_path) or ".", exist_ok=True)
    with open(config.output_path, "wb") as f:
        f.write(serialized)

    size_mb = os.path.getsize(config.output_path) / (1024 * 1024)
    print(f"엔진 저장: {config.output_path} ({size_mb:.1f} MB)")

    # 검증
    import torch
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()

    in_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)
    context.set_input_shape(in_name, (1,) + spatial)
    out_shape = tuple(context.get_tensor_shape(out_name))

    dummy_in = torch.randn(1, *spatial, dtype=torch.float32, device="cuda")
    dummy_out = torch.empty(out_shape, dtype=torch.float32, device="cuda")
    context.set_tensor_address(in_name, dummy_in.data_ptr())
    context.set_tensor_address(out_name, dummy_out.data_ptr())
    context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    torch.cuda.current_stream().synchronize()

    print(f"검증 성공: output={out_shape}")
    return config.output_path


def main():
    """CLI 진입점: python -m src.trt.builder"""
    import argparse

    parser = argparse.ArgumentParser(description="ONNX → TensorRT 엔진 변환")
    parser.add_argument("--onnx", default="assets/CLIP_EBC_nwpu_rmse_onnx.onnx")
    parser.add_argument("--output", default="assets/CLIP_EBC_nwpu_rmse_tensorrt.engine")
    parser.add_argument("--fp16", action="store_true", help="FP16 전체 (비권장)")
    parser.add_argument("--mixed", action="store_true", help="Mixed Precision (권장)")
    parser.add_argument("--workspace-gb", type=int, default=16)
    parser.add_argument("--max-batch", type=int, default=1024)
    parser.add_argument("--opt-batch", type=int, default=64)
    args = parser.parse_args()

    config = TRTBuildConfig(
        onnx_path=args.onnx,
        output_path=args.output,
        mixed_precision=args.mixed,
        fp16=args.fp16,
        workspace_gb=args.workspace_gb,
        max_batch_size=args.max_batch,
        opt_batch_size=args.opt_batch,
    )
    build_engine(config)


if __name__ == "__main__":
    main()
