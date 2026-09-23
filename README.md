# YOLO 기반 재활용 객체 탐지

재활용품의 상태와 포장 여부를 포함한 11개 클래스를 탐지하고, 동일한 학습 및 평가 조건에서 6개 YOLO 모델을 비교한 객체 탐지 프로젝트입니다. 데이터 검증, multilabel 기반 데이터 분할, 모델별 성능 비교, 20 cm 및 30 cm 거리에서의 로컬 추론 과정을 포함합니다.

## 탐지 클래스

| ID | Class | ID | Class |
|---:|---|---:|---|
| 0 | `can_normal` | 6 | `pet_dirty` |
| 1 | `can_dirty` | 7 | `pet_packaging` |
| 2 | `glass_normal` | 8 | `pet_dirty_packaging` |
| 3 | `glass_dirty` | 9 | `plastic_normal` |
| 4 | `glass_packaging` | 10 | `plastic_dirty` |
| 5 | `pet_normal` |  |  |

## 데이터셋

전처리 후 학습 데이터 122,907장, 검증 데이터 7,683장, 테스트 데이터 7,680장을 사용했습니다. `scripts/`에는 다음 작업을 위한 도구가 포함되어 있습니다.

- 누락된 image 및 잘못된 label 검사
- 클래스별 분포 집계
- 탐지 대상 클래스가 포함된 데이터 선별
- multilabel stratification 기반 validation/test 분할
- YOLO bounding box 시각화 및 샘플 검수

원본 image와 label, 로컬 manifest, PC별 절대경로는 저장 용량과 재배포 제한을 고려해 공개하지 않습니다. 데이터셋 설정은 `configs/dataset_11class.example.yaml`을 복사해 `configs/dataset_11class.yaml`로 만든 뒤 로컬 경로를 입력하면 됩니다.

## 모델 학습 및 비교

모든 모델은 `image size 640`, `batch size 16`, `optimizer SGD`, `seed 42`로 최대 30 epochs 동안 학습했습니다. 아래 값은 각 모델의 validation 결과 중 최고 성능입니다.

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLO11n | 0.7777 | 0.7986 | 0.8383 | 0.8006 |
| YOLO11s | 0.8096 | 0.8282 | 0.8687 | 0.8409 |
| YOLO12n | 0.7753 | 0.8086 | 0.8415 | 0.8057 |
| YOLO12s | 0.8028 | 0.8361 | 0.8662 | 0.8384 |
| YOLO26n | 0.7890 | 0.8006 | 0.8440 | 0.8081 |
| **YOLO26s** | **0.8048** | **0.8369** | **0.8707** | **0.8447** |

YOLO26s가 가장 높은 `mAP50`과 `mAP50-95`를 기록했습니다. Nano 모델은 상대적으로 작은 model artifact를 제공하므로 향후 edge device 배포 시 정확도와 추론 속도의 trade-off를 비교할 수 있습니다. 학습된 weight와 export한 ONNX 파일은 저장소에 포함하지 않았습니다.

## PC 추론 비교

`compare_6models_pt_20cm_30cm.py`는 6개 `.pt` 모델을 한 번씩 로드한 뒤 20 cm 및 30 cm 거리에서 촬영한 image에 추론을 수행합니다. Annotated image와 detection CSV는 `pc_compare/results/`에 저장됩니다.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:RECYCLING_MODEL_ROOT = "D:\path\to\recycling-yolo-comparison"
python compare_6models_pt_20cm_30cm.py
```

`RECYCLING_MODEL_ROOT`에는 각 학습 run directory와 `weights/best.pt`가 있어야 합니다. CUDA를 사용하지 않는 환경에서는 `RECYCLING_DEVICE=cpu`로 설정할 수 있습니다.

## 프로젝트 구조

```text
configs/                    데이터셋 설정 예시
scripts/                    데이터 전처리 및 검증 도구
reports/                    데이터셋 및 클래스 분포 요약
pc_compare/input/           거리별 비교 입력 image
pc_compare/results/         탐지 결과 image 및 CSV
compare_6models_pt_20cm_30cm.py
```

## Edge 배포 진행 상황

배포 실험을 위한 ONNX export까지 완료했습니다. Jetson Nano 배포와 device latency/FPS benchmark는 진행 중이며, 아직 실제 device에서 검증하지 않은 Edge AI 성능을 완료된 결과로 제시하지 않습니다.
