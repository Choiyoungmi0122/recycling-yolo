from pathlib import Path
import csv
import os
import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = PROJECT_ROOT / "pc_compare"
INPUT_DIR = BASE_DIR / "input"
RESULT_ROOT = BASE_DIR / "results"
MODEL_ROOT = Path(os.environ.get("RECYCLING_MODEL_ROOT", PROJECT_ROOT / "models"))

TEST_IMAGES = {
    "20cm": INPUT_DIR / "comparison_input_20.jpg",
    "30cm": INPUT_DIR / "comparison_input_30.jpg",
}

MODELS = {
    "yolo11n": MODEL_ROOT / "yolo11n_common30" / "weights" / "best.pt",
    "yolo11s": MODEL_ROOT / "yolo11s_common_v1" / "weights" / "best.pt",
    "yolo12n": MODEL_ROOT / "yolo12n_common_v1" / "weights" / "best.pt",
    "yolo12s": MODEL_ROOT / "yolo12s_common_v1" / "weights" / "best.pt",
    "yolo26n": MODEL_ROOT / "yolo26n_common30-2" / "weights" / "best.pt",
    "yolo26s": MODEL_ROOT / "yolo26s_common30-2" / "weights" / "best.pt",
}

IMGSZ = 640
CONF = 0.25
IOU = 0.45
DEVICE = os.environ.get("RECYCLING_DEVICE", "0")

for label, image_path in TEST_IMAGES.items():
    if not image_path.exists():
        raise FileNotFoundError(f"{label} input image not found: {image_path}")

for name, model_path in MODELS.items():
    if not model_path.exists():
        raise FileNotFoundError(f"{name} model not found: {model_path}")

# Load each .pt model once
loaded_models = {name: YOLO(str(path)) for name, path in MODELS.items()}

for distance, input_image in TEST_IMAGES.items():
    result_dir = RESULT_ROOT / distance
    result_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []

    print("\n" + "=" * 72)
    print(f"PC .pt comparison - {distance}")
    print(f"Input: {input_image}")
    print(f"imgsz={IMGSZ}, conf={CONF}, iou={IOU}, device={DEVICE}")
    print("=" * 72)

    for model_name, model in loaded_models.items():
        print(f"\n[{distance} / {model_name}]")

        results = model.predict(
            source=str(input_image),
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            device=DEVICE,
            verbose=False,
        )

        result = results[0]
        plotted = result.plot()

        output_image = result_dir / f"{model_name}_pt_detected.jpg"
        cv2.imwrite(str(output_image), plotted)

        boxes = result.boxes
        detection_count = 0 if boxes is None else len(boxes)

        print(f"Detections: {detection_count}")
        print(f"Saved: {output_image}")

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                cls_id = int(classes[i])
                cls_name = result.names[cls_id]
                confidence = float(confs[i])
                x1, y1, x2, y2 = [float(v) for v in xyxy[i]]

                print(
                    f"  - {cls_name}: {confidence:.3f} "
                    f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
                )

                csv_rows.append({
                    "distance": distance,
                    "model": model_name,
                    "detection_count": detection_count,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(confidence, 6),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                })
        else:
            csv_rows.append({
                "distance": distance,
                "model": model_name,
                "detection_count": 0,
                "class_id": "",
                "class_name": "",
                "confidence": "",
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
            })

    csv_path = result_dir / "pt_detection_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "distance",
                "model",
                "detection_count",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n{distance} completed")
    print(f"Results folder: {result_dir}")
    print(f"CSV: {csv_path}")

print("\nAll tests completed.")
