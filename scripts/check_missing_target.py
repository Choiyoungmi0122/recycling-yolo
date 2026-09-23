from pathlib import Path
import json


LABEL_ROOT = Path(
    "E:/recycling_data/raw/labels/train"
)


# ============================================================
# 최종 사용할 AI Hub code -> YOLO 11-class
# ============================================================

CLASS_MAP = {
    # CAN
    "c_3": "can_normal",
    "c_3_01": "can_dirty",

    # GLASS - NORMAL
    "c_4_01_02": "glass_normal",
    "c_4_02_01_02": "glass_normal",
    "c_4_02_02_02": "glass_normal",
    "c_4_02_03_02": "glass_normal",
    "c_4_03": "glass_normal",

    # GLASS - DIRTY
    "c_4_03_01": "glass_dirty",

    # GLASS - PACKAGING
    "c_4_01_01": "glass_packaging",
    "c_4_02_01_01": "glass_packaging",
    "c_4_02_02_01": "glass_packaging",
    "c_4_02_03_01": "glass_packaging",

    # PET
    "c_5_02": "pet_normal",
    "c_5_02_01": "pet_dirty",
    "c_5_01": "pet_packaging",
    "c_5_01_01": "pet_dirty_packaging",

    # PLASTIC
    "c_6": "plastic_normal",
    "c_6_01": "plastic_dirty",
}


# ============================================================
# 현재 이미지가 누락된 5개
# ============================================================

MISSING = [
    "c_20220715_000141",
    "c_20220817_003096",
    "c_20220908_004734",
    "c_20220922_003347",
    "c_20221020_003037",
]


# JSON index
json_index = {
    p.stem.lower(): p
    for p in LABEL_ROOT.rglob("*.json")
}


print("=" * 80)
print("MISSING IMAGE TARGET CHECK")
print("=" * 80)


target_missing_count = 0


for stem in MISSING:

    print()
    print("-" * 80)
    print("IMAGE:", stem)

    json_path = json_index.get(stem.lower())

    if json_path is None:
        print("JSON 없음")
        continue

    with open(
        json_path,
        "r",
        encoding="utf-8-sig",
    ) as f:
        data = json.load(f)

    original_classes = [
        obj.get("class_name")
        for obj in data.get("objects", [])
        if obj.get("class_name")
    ]

    target_classes = [
        c
        for c in original_classes
        if c in CLASS_MAP
    ]

    mapped_classes = [
        CLASS_MAP[c]
        for c in target_classes
    ]

    print("AI Hub 전체 라벨 :", original_classes)
    print("사용 대상 code   :", target_classes)
    print("YOLO class       :", mapped_classes)

    if target_classes:
        target_missing_count += 1
        print("판정             : ★ 이번 학습에 필요한 이미지")
    else:
        print("판정             : 제외 대상 → 누락되어도 문제 없음")


print()
print("=" * 80)
print(f"누락 5개 중 실제 학습 대상: {target_missing_count}개")
print("=" * 80)