"""
방법 1 통합 추론 영상 테스트 (voting 없음, frame 단위 raw 결과)

    python scripts\\test_method1_integrated.py --device 1

출력 (--output):
    method1_frame_results.csv
    method1_annotated.mp4
    method1_summary.json / method1_summary.txt
"""

from pathlib import Path
from collections import Counter
import argparse
import csv
import json
import time

import cv2
import torch

from method1_integrated_inference import (
    DEFAULT_YOLO_MODEL,
    FINAL_CLASSES,
    Method1Predictor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE = PROJECT_ROOT / "videos" / "input" / "comparison_video_30cm.mp4"
DEFAULT_OUTPUT = PROJECT_ROOT / "videos" / "30cm" / "method1_integrated"

MATERIAL_COLORS = {
    "CAN": (0, 165, 255),
    "GLASS": (255, 200, 0),
    "PET": (0, 220, 0),
    "PLASTIC": (220, 0, 220),
}

CSV_COLUMNS = [
    "frame",
    "material_id",
    "material",
    "yolo_confidence",
    "state",
    "cnn_confidence",
    "final_class_id",
    "final_class",
    "x1",
    "y1",
    "x2",
    "y2",
    "num_detections",
    "yolo_ms",
    "crop_ms",
    "cnn_ms",
    "total_ms",
]


# ============================================================
# 시각화
# ============================================================

def draw_text_block(image, lines, origin, color, scale=0.8, thickness=2):

    x, y = origin

    for line in lines:

        (tw, th), baseline = cv2.getTextSize(
            line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )

        cv2.rectangle(
            image,
            (x, y - th - 6),
            (x + tw + 8, y + baseline),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            image, line, (x + 4, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA,
        )

        y += th + baseline + 10


def annotate(frame, frame_index, result):

    if not result["detected"]:

        draw_text_block(
            frame,
            [f"frame {frame_index}", "NO DETECTION"],
            (10, 40),
            (0, 0, 255),
        )

        return

    color = MATERIAL_COLORS.get(result["material"], (255, 255, 255))

    x1, y1, x2, y2 = (int(round(v)) for v in result["bbox"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    lines = [f"YOLO: {result['material']} {result['yolo_confidence']:.2f}"]

    if result["final_class"] is not None:
        lines += [
            f"CNN: {result['state']} {result['cnn_confidence']:.2f}",
            f"FINAL: [{result['final_class_id']}] {result['final_class']}",
        ]
    else:
        lines += ["CNN: invalid crop"]

    # bbox 위쪽에 표시, 공간이 없으면 bbox 안쪽 위
    text_y = y1 - 90 if y1 - 90 > 30 else y1 + 30
    draw_text_block(frame, lines, (x1, text_y), color)

    draw_text_block(
        frame,
        [
            f"frame {frame_index}",
            f"yolo {result['yolo_ms']:.1f}ms  cnn {result['cnn_ms']:.1f}ms  "
            f"total {result['total_ms']:.1f}ms",
        ],
        (10, 40),
        (255, 255, 255),
        scale=0.7,
    )


# ============================================================
# Main
# ============================================================

def fmt(value, digits=6):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return value


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--yolo", type=str, default=str(DEFAULT_YOLO_MODEL))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--yolo-fp16", action="store_true",
                        help="YOLO FP16 (Ultralytics quantize=16). 기본 FP32")
    parser.add_argument("--no-cnn-amp", action="store_true",
                        help="CNN autocast FP16 끄기 (FP32)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="앞에서부터 N frame 만 처리 (0 = 전체)")
    parser.add_argument("--debug-timing", action="store_true",
                        help="frame 마다 YOLO / crop / CNN / total ms 출력")
    parser.add_argument("--no-save-video", action="store_true",
                        help="annotated 영상 drawing/저장 끄기 (순수 추론 속도 확인용)")
    args = parser.parse_args()

    source_path = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise FileNotFoundError(f"Video 없음:\n{source_path}")

    # --------------------------------------------------------
    # Predictor
    # --------------------------------------------------------

    predictor = Method1Predictor(
        yolo_model=args.yolo,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        yolo_fp16=args.yolo_fp16,
        cnn_amp=not args.no_cnn_amp,
    )

    print()
    print("=" * 80)
    print("Method 1: YOLO26s 4-class + EfficientNet-B0 x4 → 11-class")
    print("=" * 80)
    print(f"GPU: {args.device} - {torch.cuda.get_device_name(args.device)}")
    print(f"YOLO: {args.yolo}")
    print(f"YOLO names: {predictor.yolo_names}")
    print(f"Source: {source_path}")
    print(
        f"conf: {args.conf} | imgsz: {args.imgsz} | "
        f"YOLO: {'FP16' if args.yolo_fp16 else 'FP32'} | "
        f"CNN autocast FP16: {not args.no_cnn_amp}"
    )
    print(
        f"max frames: {args.max_frames or 'all'} | "
        f"save video: {not args.no_save_video} | debug timing: {args.debug_timing}"
    )
    print()
    print("재질 → CNN index → 최종 11-class")

    for material, cnn_index, state, final_id, final_name in predictor.mapping_table():
        print(f"  {material:8s} [{cnn_index}] {state:16s} → {final_id:2d} {final_name}")

    # --------------------------------------------------------
    # Video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(source_path))

    if not cap.isOpened():
        raise RuntimeError(f"Video 열기 실패:\n{source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    predictor.warmup(frame_shape=(height, width, 3))

    # YOLO 는 첫 predict(warmup) 이후에 device 로 옮겨지므로 여기서 확인
    print()
    print("Model devices")

    devices = predictor.device_report()

    for name, info in devices.items():
        print(f"  {name:12s}: {info}")

    expected = str(predictor.device)

    wrong = [name for name, info in devices.items() if not info.startswith(expected)]

    if wrong:
        raise RuntimeError(f"{expected} 가 아닌 device 에 올라간 모델: {wrong}")

    print()

    video_path = output_dir / "method1_annotated.mp4"
    csv_path = output_dir / "method1_frame_results.csv"

    writer = None

    if not args.no_save_video:
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    csv_file = csv_path.open("w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(CSV_COLUMNS)

    frame_index = 0
    detection_frames = 0
    classified_frames = 0

    yolo_ms_all = []
    crop_ms_classified = []
    cnn_ms_classified = []
    total_ms_all = []

    material_counts = Counter()
    final_counts = Counter()

    wall_start = time.perf_counter()

    while True:

        if args.max_frames and frame_index >= args.max_frames:
            break

        ret, frame = cap.read()

        if not ret:
            break

        frame_index += 1

        result = predictor.predict(frame)

        if args.debug_timing:
            print(
                f"frame {frame_index}  "
                f"[{result['material'] or '-'} → {result['final_class'] or '-'}]\n"
                f"  YOLO: {result['yolo_ms']:.2f} ms\n"
                f"  crop/preprocess: {result['crop_ms']:.2f} ms\n"
                f"  CNN: {result['cnn_ms']:.2f} ms\n"
                f"  total: {result['total_ms']:.2f} ms"
            )

        yolo_ms_all.append(result["yolo_ms"])
        total_ms_all.append(result["total_ms"])

        if result["detected"]:
            detection_frames += 1
            material_counts[result["material"]] += 1

        if result["final_class"] is not None:
            classified_frames += 1
            crop_ms_classified.append(result["crop_ms"])
            cnn_ms_classified.append(result["cnn_ms"])
            final_counts[result["final_class"]] += 1

        bbox = result["bbox"] or [None] * 4

        csv_writer.writerow([
            frame_index,
            fmt(result["material_id"]),
            fmt(result["material"]),
            fmt(result["yolo_confidence"]),
            fmt(result["state"]),
            fmt(result["cnn_confidence"]),
            fmt(result["final_class_id"]),
            fmt(result["final_class"]),
            *(fmt(v, 1) for v in bbox),
            result["num_detections"],
            fmt(result["yolo_ms"], 3),
            fmt(result["crop_ms"], 3),
            fmt(result["cnn_ms"], 3),
            fmt(result["total_ms"], 3),
        ])

        if writer is not None:
            annotate(frame, frame_index, result)
            writer.write(frame)

        if frame_index % 100 == 0:
            print(
                f"  frame {frame_index} | det {detection_frames} | "
                f"avg total {mean(total_ms_all):.1f}ms"
            )

    wall_seconds = time.perf_counter() - wall_start

    cap.release()
    if writer is not None:
        writer.release()
    csv_file.close()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_frames = frame_index

    summary = {
        "source": str(source_path),
        "yolo_model": str(args.yolo),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(args.device),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "yolo_fp16": args.yolo_fp16,
        "cnn_autocast_fp16": not args.no_cnn_amp,
        "max_frames": args.max_frames,
        "total_frames": total_frames,
        "yolo_detection_frames": detection_frames,
        "yolo_detection_rate": detection_frames / total_frames if total_frames else 0.0,
        "cnn_classified_frames": classified_frames,
        "cnn_classified_rate": classified_frames / total_frames if total_frames else 0.0,
        # YOLO / total: 전체 frame 평균, CNN: CNN 실행된 frame 평균
        "avg_yolo_ms": mean(yolo_ms_all),
        "avg_crop_ms": mean(crop_ms_classified),
        "avg_cnn_ms": mean(cnn_ms_classified),
        "avg_total_ms": mean(total_ms_all),
        # 추론만 (predict 호출 시간 합 기준)
        "processing_fps": 1000.0 / mean(total_ms_all) if total_ms_all else 0.0,
        # 영상 decode + CSV (+ 시각화/저장, --no-save-video 가 아니면) 포함
        "end_to_end_fps": total_frames / wall_seconds if wall_seconds else 0.0,
        "material_frame_counts": {
            name: material_counts.get(name, 0)
            for name in predictor.yolo_names.values()
        },
        "final_class_frame_counts": {
            f"{index} {name}": final_counts.get(name, 0)
            for index, name in enumerate(FINAL_CLASSES)
        },
        "outputs": {
            "csv": str(csv_path),
            "video": None if args.no_save_video else str(video_path),
        },
    }

    json_path = output_dir / "method1_summary.json"
    txt_path = output_dir / "method1_summary.txt"

    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "Method 1 Integrated Inference Summary",
        "=" * 60,
        f"total frames            : {total_frames}",
        f"YOLO detection frames   : {detection_frames} ({summary['yolo_detection_rate']:.2%})",
        f"CNN classified frames   : {classified_frames} ({summary['cnn_classified_rate']:.2%})",
        f"avg YOLO latency        : {summary['avg_yolo_ms']:.2f} ms (all frames)",
        f"avg crop/preprocess     : {summary['avg_crop_ms']:.2f} ms (classified frames)",
        f"avg CNN latency         : {summary['avg_cnn_ms']:.2f} ms (classified frames)",
        f"avg total latency       : {summary['avg_total_ms']:.2f} ms (all frames)",
        f"processing FPS          : {summary['processing_fps']:.2f} (inference only)",
        f"end-to-end FPS          : {summary['end_to_end_fps']:.2f} (incl. decode/CSV/draw/write)",
        "",
        "YOLO material frames",
        *(f"  {name:8s}: {count}" for name, count in summary["material_frame_counts"].items()),
        "",
        "Final 11-class frames",
        *(f"  {name:22s}: {count}" for name, count in summary["final_class_frame_counts"].items()),
    ]

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("\n".join(lines))
    print()
    print(f"CSV    : {csv_path}")
    print(f"Video  : {'(saved off)' if args.no_save_video else video_path}")
    print(f"Summary: {json_path}")
    print(f"         {txt_path}")


if __name__ == "__main__":
    main()
