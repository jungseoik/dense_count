# TensorRT FP32 변환 및 추론 가이드

## 개요

CLIP-EBC ONNX 모델을 TensorRT FP32 엔진으로 변환하여 추론 속도를 향상시킵니다.
ONNX 대비 약 1.4배 빠르며, 정확도 손실은 거의 없습니다 (count diff < 0.05).

## 사전 요구사항

```bash
conda activate ebc
# 필수 패키지: torch, tensorrt, onnxruntime-gpu
```

## 1. 엔진 빌드

### Python API

```python
from src.config import TRTBuildConfig
from src.trt import build_engine

config = TRTBuildConfig(
    onnx_path="assets/CLIP_EBC_nwpu_rmse_onnx.onnx",
    output_path="assets/CLIP_EBC_nwpu_rmse_tensorrt.engine",
    mixed_precision=False,   # FP32 모드
    fp16=False,
    workspace_gb=16,         # GPU에 맞게 조정
    max_batch_size=1024,     # 동시 처리 타일 수 상한
    opt_batch_size=64,       # 가장 많이 사용할 배치 크기
)

engine_path = build_engine(config)
```

### CLI

```bash
python -m src.trt.builder --output assets/CLIP_EBC_nwpu_rmse_tensorrt.engine
```

### 주요 파라미터

| 파라미터 | 설명 | 기본값 |
|---|---|---|
| `max_batch_size` | 한번에 추론 가능한 최대 타일 수. FHD=45tiles, 4K=~180tiles | 1024 |
| `opt_batch_size` | TRT가 최적화하는 타겟 배치 크기 | 64 |
| `workspace_gb` | 엔진 빌드 시 GPU workspace. 클수록 더 좋은 tactic 선택 | 16 |

## 2. 추론

```python
from src.config import InferenceConfig
from src.trt import ClipEBCTensorRT
import numpy as np

config = InferenceConfig(model_path="assets/CLIP_EBC_nwpu_rmse_tensorrt.engine")
model = ClipEBCTensorRT(config)

# 배치 추론 (여러 이미지)
images = [np.array(img1), np.array(img2)]  # List[np.ndarray], (H,W,3) uint8
counts = model.predict(images)  # [221.49, 553.10]

# 단일 이미지
count = model.predict_single("path/to/image.jpg")  # 221.49
```

## 3. 성능 참고 (RTX PRO 6000, 97GB)

| 해상도 | tiles | ONNX | TRT FP32 | 속도비 |
|---|---|---|---|---|
| SD (640x360) | 6 | 7.2ms | 5.0ms | 1.4x |
| HD (1280x720) | 24 | 25.7ms | 18.2ms | 1.4x |
| FHD (1920x1080) | 45 | 48.7ms | 34.6ms | 1.4x |

## 4. 주의사항

- TRT 엔진은 빌드한 GPU 아키텍처에 종속됨. GPU 교체 시 재빌드 필요.
- `max_batch_size`를 크게 잡으면 엔진 로드 시 VRAM 소비 증가.
- FP32이므로 ONNX와 거의 동일한 정확도 (diff < 0.05).
