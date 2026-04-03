from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """ONNX / TensorRT 추론 공통 설정."""
    model_path: str
    window_size: int = 224
    stride: int = 224
    reduction: int = 8
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


@dataclass
class TRTBuildConfig:
    """TensorRT 엔진 빌드 설정."""
    onnx_path: str = "assets/CLIP_EBC_nwpu_rmse_onnx.onnx"
    output_path: str = "assets/CLIP_EBC_nwpu_rmse_tensorrt.engine"
    mixed_precision: bool = True
    fp16: bool = False
    workspace_gb: int = 16
    max_batch_size: int = 1024
    opt_batch_size: int = 64
