# TensorRT Mixed Precision 변환 및 추론 가이드

## 개요

Mixed Precision은 대부분의 레이어를 FP16으로 실행하되,
수치적으로 민감한 레이어(Softmax, L2 Norm, Exp, ReduceSum)만 FP32로 고정합니다.

- ONNX 대비 약 **2.5배** 빠름
- 엔진 크기 **절반** (372MB -> 188MB)
- VRAM 사용량 **절반**
- 정확도: count diff < 1.0, 상관계수 > 0.999

## FP16 전체 모드를 사용하지 않는 이유

이 CLIP-EBC 모델은 FP16 전체 변환 시 ViT Attention 내부에서 정밀도 누적 오류가 발생하여
결과가 완전히 깨집니다 (count 221 -> 5395). Mixed Precision이 유일한 FP16 활용 방법입니다.

### FP32로 고정되는 레이어 (32개)

- Softmax (13개): Attention Softmax 12개 + 최종 분류 Softmax 1개
- ReduceL2: L2 정규화
- Exp: logit_scale 지수 연산
- ReduceSum: 최종 density 합산

## 1. 엔진 빌드

### Python API

```python
from src.config import TRTBuildConfig
from src.trt import build_engine

config = TRTBuildConfig(
    onnx_path="assets/CLIP_EBC_nwpu_rmse_onnx.onnx",
    output_path="assets/CLIP_EBC_nwpu_rmse_tensorrt_mixed.engine",
    mixed_precision=True,    # Mixed Precision 활성화
    workspace_gb=16,
    max_batch_size=1024,
    opt_batch_size=64,
)

engine_path = build_engine(config)
```

### CLI

```bash
python -m src.trt.builder --mixed --output assets/CLIP_EBC_nwpu_rmse_tensorrt_mixed.engine
```

## 2. 추론

```python
from src.config import InferenceConfig
from src.trt import ClipEBCTensorRT

config = InferenceConfig(model_path="assets/CLIP_EBC_nwpu_rmse_tensorrt_mixed.engine")
model = ClipEBCTensorRT(config)

# 배치 추론
counts = model.predict([img1, img2, img3])

# 단일 이미지
count = model.predict_single("assets/289.jpg")
```

## 3. 성능 비교 (RTX PRO 6000, 97GB)

### 추론 속도 (순수 추론, 전처리 제외)

| 해상도 | tiles | ONNX | FP32 | Mixed | Mixed 속도비 |
|---|---|---|---|---|---|
| SD (640x360) | 6 | 7.2ms | 5.0ms | 2.5ms | **2.9x** |
| HD (1280x720) | 24 | 25.7ms | 18.2ms | 9.9ms | **2.6x** |
| FHD (1920x1080) | 45 | 48.7ms | 34.6ms | 18.8ms | **2.6x** |

### VRAM 사용량

| | 모델 로드 | 추론 중 추가 (FHD) |
|---|---|---|
| ONNX | 2.5 GB | +2.6 GB |
| TRT FP32 | 13.8 GB | +0 MB |
| TRT Mixed | **6.5 GB** | **+0 MB** |

### 정확도

| | count diff (최대) | 상관계수 (최소) | density MAE |
|---|---|---|---|
| TRT FP32 | 0.03 | 0.999999 | < 0.00005 |
| TRT Mixed | 0.92 | 0.999971 | < 0.00026 |

## 4. GPU 교체 시 재빌드 방법

```python
from src.trt import build_engine
from src.config import TRTBuildConfig

# GPU에 맞게 workspace 조정
build_engine(TRTBuildConfig(
    mixed_precision=True,
    workspace_gb=8,       # VRAM이 적은 GPU에서는 줄임
    max_batch_size=256,   # VRAM에 맞게 조정
))
```

## 5. 벤치마크 실행

GPU 교체 후 성능 확인:

```bash
# VRAM + 속도 벤치마크
python -m src.tests.test_vram_benchmark --models trt_mixed

# 정확도 비교
python -m src.tests.test_accuracy
```
