# YOLO-Based Recycling Object Detection

An 11-class recycling object detection project that compares six YOLO model variants under a common training and evaluation setup. The project covers dataset validation, multilabel train/validation/test preparation, model comparison, and local inference at two camera distances.

## Classes

| ID | Class | ID | Class |
|---:|---|---:|---|
| 0 | `can_normal` | 6 | `pet_dirty` |
| 1 | `can_dirty` | 7 | `pet_packaging` |
| 2 | `glass_normal` | 8 | `pet_dirty_packaging` |
| 3 | `glass_dirty` | 9 | `plastic_normal` |
| 4 | `glass_packaging` | 10 | `plastic_dirty` |
| 5 | `pet_normal` |  |  |

## Dataset

The prepared dataset uses 122,907 training images, 7,683 validation images, and 7,680 test images. Utilities in `scripts/` audit missing files and malformed labels, summarize class distributions, filter target classes, and produce a multilabel-stratified validation/test split.

The original images, labels, local manifests, and machine-specific paths are not included in this repository because of storage and redistribution constraints. Start from `configs/dataset_11class.example.yaml` and create a local `configs/dataset_11class.yaml`.

## Model Comparison

All models were trained for up to 30 epochs with a 640-pixel image size, batch size 16, SGD, and seed 42. The table reports the best validation result from each run.

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLO11n | 0.7777 | 0.7986 | 0.8383 | 0.8006 |
| YOLO11s | 0.8096 | 0.8282 | 0.8687 | 0.8409 |
| YOLO12n | 0.7753 | 0.8086 | 0.8415 | 0.8057 |
| YOLO12s | 0.8028 | 0.8361 | 0.8662 | 0.8384 |
| YOLO26n | 0.7890 | 0.8006 | 0.8440 | 0.8081 |
| **YOLO26s** | **0.8048** | **0.8369** | **0.8707** | **0.8447** |

YOLO26s achieved the highest mAP50 and mAP50-95, while the nano variants provide smaller deployment artifacts. Exported ONNX models are kept outside the repository.

## Local Inference

`compare_6models_pt_20cm_30cm.py` loads the six trained models once, runs inference on 20 cm and 30 cm samples, and saves annotated images and detection CSV files under `pc_compare/results/`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:RECYCLING_MODEL_ROOT = "D:\path\to\recycling-yolo-comparison"
python compare_6models_pt_20cm_30cm.py
```

The model root should contain each training run directory and its `weights/best.pt` file. Set `RECYCLING_DEVICE=cpu` to run without CUDA.

## Project Structure

```text
configs/                    Dataset configuration example
scripts/                    Dataset preparation and validation utilities
reports/                    Compact dataset and class summaries
pc_compare/input/           Local comparison samples
pc_compare/results/         Detection images and CSV outputs
compare_6models_pt_20cm_30cm.py
```

## Deployment Status

ONNX exports have been prepared for deployment experiments. Jetson Nano deployment and device-side latency/FPS benchmarking are in progress; no completed edge benchmark is claimed yet.

