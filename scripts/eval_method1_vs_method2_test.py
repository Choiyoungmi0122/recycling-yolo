"""
방법 1 (YOLO26s 4-class + 재질별 EfficientNet-B0) vs 방법 2 (YOLO26s 11-class)
test set 최종 11-class 성능 비교

같은 이미지 / 같은 계산 코드로 두 방법을 평가한다.

1) 검출 지표 (Ultralytics val 과 같은 방식)
   - conf 0.001 / max_det 300 으로 예측, IoU 0.50:0.95 매칭, ap_per_class
   - 방법 1 은 4-class bbox 를 CNN 결과로 11-class 로 바꾸고
     confidence = YOLO conf × CNN conf 로 사용
   → P / R / mAP50 / mAP50-95 (전체 + class 별)

2) 주 객체 판정 정확도 (실제 시스템 방식)
   - conf 0.25, 이미지마다 면적이 가장 큰 예측 bbox 1개 = 시스템 판정
   - 정답 = 면적이 가장 큰 GT bbox 의 11-class
   - 정답 조건: class 일치 + 두 bbox IoU ≥ 0.5
   → 11-class 정확도, 재질(4-class) 정확도, 미검출률, class 별 recall, confusion matrix

실행:
    python scripts\\eval_method1_vs_method2_test.py --device 1
    python scripts\\eval_method1_vs_method2_test.py --device 1 --limit 200   (빠른 확인)
"""

from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
import json
import time

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class

from method1_integrated_inference import (
    CNN_TRANSFORM,
    DEFAULT_CNN_MODELS,
    DEFAULT_YOLO_MODEL,
    FINAL_CLASSES,
    load_state_classifier,
    padded_crop_box,
)


DEFAULT_YOLO11 = Path(
    r"E:\recycling_data\runs\recycling-yolo-comparison\yolo26s_common30-2\weights\best.pt"
)
DEFAULT_LIST = Path(r"E:\recycling_data\test_final.txt")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "reports" / "test_eval" / "method1_vs_method2"

IOUV = np.linspace(0.5, 0.95, 10)
NC = len(FINAL_CLASSES)
MATERIALS = ["CAN", "GLASS", "PET", "PLASTIC"]


def final_to_material(class_id):
    return MATERIALS.index(FINAL_CLASSES[class_id].split("_", 1)[0].upper())


# ============================================================
# 데이터
# ============================================================

def label_path(image_path):
    p = Path(image_path)
    parts = list(p.parts)
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_gt(image_path, width, height):
    """YOLO txt (11-class) → classes (N,), boxes xyxy pixel (N, 4)"""

    path = label_path(image_path)
    rows = []

    if path.exists():
        for line in path.read_text().splitlines():
            v = line.split()
            if len(v) == 5:
                rows.append([float(x) for x in v])

    if not rows:
        return np.zeros(0, int), np.zeros((0, 4))

    a = np.array(rows)
    cls = a[:, 0].astype(int)
    cx, cy, bw, bh = a[:, 1] * width, a[:, 2] * height, a[:, 3] * width, a[:, 4] * height
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)

    return cls, boxes


def box_iou(a, b):
    """a (N,4), b (M,4) → (N,M)"""

    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))

    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)

    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_predictions(pred_cls, gt_cls, iou):
    """Ultralytics BaseValidator.match_predictions 와 동일 (greedy). iou: (gt, pred)"""

    correct = np.zeros((len(pred_cls), len(IOUV)), bool)
    iou = iou * (gt_cls[:, None] == pred_cls[None, :])

    for i, threshold in enumerate(IOUV):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True

    return correct


# ============================================================
# 예측
# ============================================================

def yolo_boxes(model, image, device, conf, imgsz):

    r = model.predict(image, imgsz=imgsz, conf=conf, max_det=300, device=device, verbose=False)[0]

    return (
        r.boxes.cls.cpu().numpy().astype(int),
        r.boxes.conf.cpu().numpy(),
        r.boxes.xyxy.cpu().numpy(),
    )


def method1_classify(cnn, image_bgr, materials, boxes, device, amp):
    """4-class bbox 전부를 해당 재질 CNN 으로 분류 → (final_cls, cnn_conf)"""

    h, w = image_bgr.shape[:2]
    final_cls = np.full(len(boxes), -1, int)
    cnn_conf = np.zeros(len(boxes))

    for m_id, material in enumerate(MATERIALS):

        idx = np.nonzero(materials == m_id)[0]
        tensors, keep = [], []

        for i in idx:
            x1, y1, x2, y2 = padded_crop_box(boxes[i], w, h)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            tensors.append(CNN_TRANSFORM(Image.fromarray(crop)))
            keep.append(i)

        if not keep:
            continue

        model, final_ids, _ = cnn[material.lower()]
        batch = torch.stack(tensors).to(device)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            probs = torch.softmax(model(batch).float(), 1).cpu().numpy()

        best = probs.argmax(1)
        final_cls[keep] = np.array(final_ids)[best]
        cnn_conf[keep] = probs[np.arange(len(keep)), best]

    return final_cls, cnn_conf


# ============================================================
# 지표
# ============================================================

class DetStats:

    def __init__(self):
        self.tp, self.conf, self.pred_cls, self.target_cls = [], [], [], []

    def add(self, pred_cls, conf, pred_boxes, gt_cls, gt_boxes):
        self.target_cls.append(gt_cls)
        if len(pred_cls) == 0:
            return
        self.tp.append(match_predictions(pred_cls, gt_cls, box_iou(gt_boxes, pred_boxes)))
        self.conf.append(conf)
        self.pred_cls.append(pred_cls)

    def result(self):

        tp = np.concatenate(self.tp, 0)
        conf = np.concatenate(self.conf)
        pred_cls = np.concatenate(self.pred_cls)
        target_cls = np.concatenate(self.target_cls)

        _, _, p, r, _, ap, classes, *_ = ap_per_class(
            tp, conf, pred_cls, target_cls, names={i: n for i, n in enumerate(FINAL_CLASSES)},
        )

        counts = np.bincount(target_cls, minlength=NC)

        return {
            "precision": float(p.mean()),
            "recall": float(r.mean()),
            "map50": float(ap[:, 0].mean()),
            "map50_95": float(ap.mean()),
            "per_class": {
                FINAL_CLASSES[c]: {
                    "instances": int(counts[c]),
                    "p": float(p[i]), "r": float(r[i]),
                    "map50": float(ap[i, 0]), "map50_95": float(ap[i].mean()),
                }
                for i, c in enumerate(classes)
            },
        }


class PrimaryStats:
    """주 객체 판정. confusion[gt, pred], pred 열 마지막 = 미검출"""

    def __init__(self):
        self.confusion = np.zeros((NC, NC + 1), int)
        self.correct = 0
        self.material_correct = 0
        self.total = 0
        self.no_detection = 0
        self.low_iou = 0

    def add(self, gt_cls, gt_boxes, pred_cls, pred_boxes):

        if len(gt_cls) == 0:
            return None

        g = int(np.argmax((gt_boxes[:, 2:] - gt_boxes[:, :2]).prod(1)))
        self.total += 1

        if len(pred_cls) == 0:
            self.no_detection += 1
            self.confusion[gt_cls[g], NC] += 1
            return {"gt": int(gt_cls[g]), "pred": None, "iou": 0.0, "correct": False}

        k = int(np.argmax((pred_boxes[:, 2:] - pred_boxes[:, :2]).prod(1)))
        iou = float(box_iou(gt_boxes[g:g + 1], pred_boxes[k:k + 1])[0, 0])
        pc = int(pred_cls[k])

        self.confusion[gt_cls[g], pc] += 1

        located = iou >= 0.5
        ok = located and pc == gt_cls[g]

        if not located:
            self.low_iou += 1
        if ok:
            self.correct += 1
        if located and final_to_material(pc) == final_to_material(gt_cls[g]):
            self.material_correct += 1

        return {"gt": int(gt_cls[g]), "pred": pc, "iou": iou, "correct": ok}

    def result(self):

        n = max(self.total, 1)
        rows = self.confusion

        per_class = {}
        for c in range(NC):
            support = int(rows[c].sum())
            per_class[FINAL_CLASSES[c]] = {
                "support": support,
                "recall": float(rows[c, c] / support) if support else 0.0,
                "precision": float(rows[c, c] / rows[:, c].sum()) if rows[:, c].sum() else 0.0,
            }

        return {
            "images": self.total,
            "accuracy_11class": self.correct / n,
            "accuracy_material": self.material_correct / n,
            "no_detection_rate": self.no_detection / n,
            "low_iou_rate": self.low_iou / n,
            # 주 객체 GT 가 있는 class 만 평균
            "macro_recall": float(np.mean([v["recall"] for v in per_class.values() if v["support"]])),
            "per_class": per_class,
            "confusion_matrix": {
                "rows_gt": FINAL_CLASSES,
                "cols_pred": FINAL_CLASSES + ["no_detection"],
                "values": rows.tolist(),
            },
        }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default=str(DEFAULT_LIST))
    parser.add_argument("--yolo4", default=str(DEFAULT_YOLO_MODEL))
    parser.add_argument("--yolo11", default=str(DEFAULT_YOLO11))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--map-conf", type=float, default=0.001)
    parser.add_argument("--deploy-conf", type=float, default=0.25)
    parser.add_argument("--no-cnn-amp", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    amp = not args.no_cnn_amp

    images = [l.strip() for l in open(args.list, encoding="utf-8") if l.strip()]
    if args.limit:
        images = images[: args.limit]

    yolo4 = YOLO(args.yolo4)
    yolo11 = YOLO(args.yolo11)
    cnn = {m: load_state_classifier(p, m, device) for m, p in DEFAULT_CNN_MODELS.items()}

    assert [yolo4.names[i] for i in range(4)] == MATERIALS, yolo4.names
    assert [yolo11.names[i] for i in range(NC)] == FINAL_CLASSES, yolo11.names

    print(f"test images: {len(images)} | GPU {args.device} | map conf {args.map_conf} | deploy conf {args.deploy_conf}")

    det = {"method1": DetStats(), "method2": DetStats()}
    prim = {"method1": PrimaryStats(), "method2": PrimaryStats()}

    csv_file = (out / "primary_per_image.csv").open("w", newline="", encoding="utf-8-sig")
    writer = csv.writer(csv_file)
    writer.writerow(["image", "gt_class", "m1_pred", "m1_iou", "m1_correct", "m2_pred", "m2_iou", "m2_correct"])

    def read(path):
        return path, cv2.imread(path)

    def prefetch(pool, paths, depth=8):
        """앞으로 depth 장만 미리 읽음 (pool.map 은 전체를 한꺼번에 읽어 메모리 부족)"""
        pending = deque()
        for path in paths:
            pending.append(pool.submit(read, path))
            if len(pending) >= depth:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()

    t0 = time.perf_counter()

    with ThreadPoolExecutor(4) as pool:

        for n, (path, image) in enumerate(prefetch(pool, images), 1):

            if image is None:
                print(f"  읽기 실패: {path}")
                continue

            h, w = image.shape[:2]
            gt_cls, gt_boxes = load_gt(path, w, h)

            # ---- 방법 1: 4-class → CNN → 11-class
            m_cls, m_conf, m_boxes = yolo_boxes(yolo4, image, args.device, args.map_conf, args.imgsz)
            f_cls, c_conf = method1_classify(cnn, image, m_cls, m_boxes, device, amp)
            valid = f_cls >= 0
            m1_cls, m1_conf, m1_boxes = f_cls[valid], (m_conf * c_conf)[valid], m_boxes[valid]
            m1_yolo_conf = m_conf[valid]

            # ---- 방법 2: 11-class 직접
            m2_cls, m2_conf, m2_boxes = yolo_boxes(yolo11, image, args.device, args.map_conf, args.imgsz)

            det["method1"].add(m1_cls, m1_conf, m1_boxes, gt_cls, gt_boxes)
            det["method2"].add(m2_cls, m2_conf, m2_boxes, gt_cls, gt_boxes)

            # 주 객체: 실제 시스템처럼 YOLO conf ≥ deploy_conf 인 bbox 중 면적 최대
            k1 = m1_yolo_conf >= args.deploy_conf
            k2 = m2_conf >= args.deploy_conf
            r1 = prim["method1"].add(gt_cls, gt_boxes, m1_cls[k1], m1_boxes[k1])
            r2 = prim["method2"].add(gt_cls, gt_boxes, m2_cls[k2], m2_boxes[k2])

            if r1 is not None:
                name = lambda c: "" if c is None else FINAL_CLASSES[c]  # noqa: E731
                writer.writerow([
                    path, FINAL_CLASSES[r1["gt"]],
                    name(r1["pred"]), f"{r1['iou']:.3f}", int(r1["correct"]),
                    name(r2["pred"]), f"{r2['iou']:.3f}", int(r2["correct"]),
                ])

            if n % 500 == 0:
                el = time.perf_counter() - t0
                print(
                    f"  {n}/{len(images)} | {n / el:.1f} img/s | "
                    f"acc m1 {prim['method1'].correct / max(prim['method1'].total, 1):.3f} "
                    f"m2 {prim['method2'].correct / max(prim['method2'].total, 1):.3f}",
                    flush=True,
                )

    csv_file.close()

    result = {
        "test_list": args.list,
        "images": len(images),
        "settings": {
            "imgsz": args.imgsz, "map_conf": args.map_conf, "deploy_conf": args.deploy_conf,
            "method1_confidence": "yolo_conf * cnn_conf", "cnn_autocast_fp16": amp,
            "primary_rule": "largest pred box (yolo conf >= deploy_conf) vs largest GT box, correct = same class and IoU >= 0.5",
        },
        "models": {"yolo4": args.yolo4, "yolo11": args.yolo11,
                   "cnn": {m: str(p) for m, p in DEFAULT_CNN_MODELS.items()}},
        "detection": {k: v.result() for k, v in det.items()},
        "primary": {k: v.result() for k, v in prim.items()},
        "seconds": time.perf_counter() - t0,
    }

    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 요약 표 (markdown)
    d1, d2 = result["detection"]["method1"], result["detection"]["method2"]
    p1, p2 = result["primary"]["method1"], result["primary"]["method2"]

    lines = [
        "# 방법 1 vs 방법 2 — test set 11-class 비교",
        "",
        f"- test 이미지: {len(images)}장 (`{args.list}`)",
        f"- 방법 1: YOLO26s 4-class + 재질별 EfficientNet-B0 (confidence = YOLO × CNN)",
        f"- 방법 2: YOLO26s 11-class",
        "",
        "## 1. 검출 지표 (conf 0.001, IoU 0.50:0.95)",
        "",
        "| 지표 | 방법 1 | 방법 2 |",
        "|---|---|---|",
        *(f"| {k} | {d1[k]:.4f} | {d2[k]:.4f} |" for k in ["precision", "recall", "map50", "map50_95"]),
        "",
        "## 2. 주 객체 판정 (conf 0.25, 면적 최대 bbox, IoU ≥ 0.5)",
        "",
        "| 지표 | 방법 1 | 방법 2 |",
        "|---|---|---|",
        f"| 11-class 정확도 | {p1['accuracy_11class']:.4f} | {p2['accuracy_11class']:.4f} |",
        f"| 재질(4-class) 정확도 | {p1['accuracy_material']:.4f} | {p2['accuracy_material']:.4f} |",
        f"| class 평균 recall | {p1['macro_recall']:.4f} | {p2['macro_recall']:.4f} |",
        f"| 미검출률 | {p1['no_detection_rate']:.4f} | {p2['no_detection_rate']:.4f} |",
        f"| 위치 불일치 (IoU<0.5) | {p1['low_iou_rate']:.4f} | {p2['low_iou_rate']:.4f} |",
        "",
        "## 3. class 별",
        "",
        "| class | GT 수 (주 객체) | mAP50-95 방법1 | mAP50-95 방법2 | 판정 recall 방법1 | 판정 recall 방법2 |",
        "|---|---|---|---|---|---|",
    ]

    for c in FINAL_CLASSES:
        a1 = d1["per_class"].get(c, {}).get("map50_95", 0.0)
        a2 = d2["per_class"].get(c, {}).get("map50_95", 0.0)
        lines.append(
            f"| {c} | {p1['per_class'][c]['support']} | {a1:.4f} | {a2:.4f} | "
            f"{p1['per_class'][c]['recall']:.4f} | {p2['per_class'][c]['recall']:.4f} |"
        )

    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
