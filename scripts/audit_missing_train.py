from pathlib import Path
import csv
import json


IMAGE_ROOT = Path(
    "E:/recycling_data/raw/images/train/application_C"
)

LABEL_ROOT = Path(
    "E:/recycling_data/raw/labels/train"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "data_audit"

REPORT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 우리가 사용하는 18개 AI Hub 코드
# ============================================================

CLASS_MAP = {
    "c_3": "can_normal",
    "c_3_01": "can_dirty",

    "c_4_01_02": "glass_normal",
    "c_4_02_01_02": "glass_normal",
    "c_4_02_02_02": "glass_normal",
    "c_4_02_03_02": "glass_normal",
    "c_4_03": "glass_normal",

    "c_4_03_01": "glass_dirty",

    "c_4_01_01": "glass_packaging",
    "c_4_02_01_01": "glass_packaging",
    "c_4_02_02_01": "glass_packaging",
    "c_4_02_03_01": "glass_packaging",

    "c_5_02": "pet_normal",
    "c_5_02_01": "pet_dirty",
    "c_5_01": "pet_packaging",
    "c_5_01_01": "pet_dirty_packaging",

    "c_6": "plastic_normal",
    "c_6_01": "plastic_dirty",
}


def classify(codes):
    has_target = any(code in CLASS_MAP for code in codes)

    if not has_target:
        return "NO_TARGET"

    if all(code in CLASS_MAP for code in codes):
        return "TARGET_ONLY"

    return "TARGET_MIXED"


def main():

    print("이미지 index 생성 중...")

    image_stems = {
        p.stem.lower()
        for p in IMAGE_ROOT.rglob("*.jpg")
    }

    print(f"이미지: {len(image_stems):,}")

    json_files = list(
        LABEL_ROOT.rglob("*.json")
    )

    print(f"JSON: {len(json_files):,}")

    rows = []

    for json_path in json_files:

        stem = json_path.stem.lower()

        # 이미지가 있으면 관심 없음
        if stem in image_stems:
            continue

        try:
            with open(
                json_path,
                "r",
                encoding="utf-8-sig",
            ) as f:
                data = json.load(f)

        except Exception as e:

            rows.append({
                "stem": stem,
                "status": "JSON_ERROR",
                "original_codes": "",
                "yolo_classes": "",
                "reason": str(e),
            })

            continue

        codes = [
            obj.get("class_name")
            for obj in data.get("objects", [])
            if obj.get("class_name")
        ]

        state = classify(codes)

        yolo_classes = sorted({
            CLASS_MAP[c]
            for c in codes
            if c in CLASS_MAP
        })

        if state == "TARGET_ONLY":
            reason = "원천 이미지 누락으로 학습 제외"
        elif state == "TARGET_MIXED":
            reason = "A 방식에서 원래 제외되는 혼합 이미지"
        else:
            reason = "11-class 비대상 이미지"

        rows.append({
            "stem": stem,
            "status": state,
            "original_codes": "|".join(codes),
            "yolo_classes": "|".join(yolo_classes),
            "reason": reason,
        })

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_path = REPORT_ROOT / "missing_train_images.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stem",
                "status",
                "original_codes",
                "yolo_classes",
                "reason",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # 실제 A 학습에서 제외해야 할 stem만 TXT로 저장
    # --------------------------------------------------------

    target_only_missing = [
        row["stem"]
        for row in rows
        if row["status"] == "TARGET_ONLY"
    ]

    exclude_path = (
        REPORT_ROOT
        / "exclude_missing_target_only.txt"
    )

    exclude_path.write_text(
        "\n".join(target_only_missing) + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 출력
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MISSING TRAIN AUDIT")
    print("=" * 70)

    for row in rows:
        print(
            row["stem"],
            "->",
            row["status"],
            row["original_codes"],
        )

    print()
    print(f"전체 누락 이미지: {len(rows)}")
    print(
        "A 학습에 실제 영향:",
        len(target_only_missing),
    )

    print()
    print("CSV :", csv_path)
    print("EXCLUDE :", exclude_path)


if __name__ == "__main__":
    main()
