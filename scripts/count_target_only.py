from pathlib import Path
from collections import Counter
import json


# ============================================================
# 경로
# ============================================================

LABEL_ROOT = Path(
    "E:/recycling_data/raw/labels/train"
)

IMAGE_ROOT = Path(
    "E:/recycling_data/raw/images/train/application_C"
)


# ============================================================
# 최종 11-class mapping
# ============================================================

CLASS_MAP = {
    # --------------------------------------------------------
    # CAN
    # --------------------------------------------------------
    "c_3": "can_normal",
    "c_3_01": "can_dirty",

    # --------------------------------------------------------
    # GLASS - NORMAL
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # PET
    # --------------------------------------------------------
    "c_5_02": "pet_normal",
    "c_5_02_01": "pet_dirty",

    "c_5_01": "pet_packaging",

    "c_5_01_01": "pet_dirty_packaging",

    # --------------------------------------------------------
    # PLASTIC
    # --------------------------------------------------------
    "c_6": "plastic_normal",
    "c_6_01": "plastic_dirty",
}


CLASS_NAMES = [
    "can_normal",
    "can_dirty",
    "glass_normal",
    "glass_dirty",
    "glass_packaging",
    "pet_normal",
    "pet_dirty",
    "pet_packaging",
    "pet_dirty_packaging",
    "plastic_normal",
    "plastic_dirty",
]


# ============================================================
# 이미지 확인
# ============================================================

print("Training 이미지 확인 중...")

image_stems = {
    p.stem.lower()
    for p in IMAGE_ROOT.rglob("*.jpg")
}

print(
    f"실제 이미지 수: "
    f"{len(image_stems):,}"
)


# ============================================================
# JSON 확인
# ============================================================

json_files = list(
    LABEL_ROOT.rglob("*.json")
)

print(
    f"JSON 수: "
    f"{len(json_files):,}"
)

print()


# ============================================================
# 통계
# ============================================================

stats = Counter()

class_image_count = Counter()
class_bbox_count = Counter()

# 참고용:
# 대상 bbox / 비대상 bbox 수
target_bbox_total = 0
ignored_bbox_total = 0


# ============================================================
# 전체 JSON 순회
# ============================================================

for idx, json_path in enumerate(
    json_files,
    start=1,
):

    if idx % 20000 == 0:
        print(
            f"처리 중: "
            f"{idx:,} / "
            f"{len(json_files):,}"
        )

    # --------------------------------------------------------
    # JSON READ
    # --------------------------------------------------------

    try:
        with open(
            json_path,
            "r",
            encoding="utf-8-sig",
        ) as f:
            data = json.load(f)

    except Exception:
        stats["JSON_ERROR"] += 1
        continue


    objects = data.get(
        "objects",
        []
    )

    if not objects:
        stats["NO_OBJECT"] += 1
        continue


    # --------------------------------------------------------
    # 이미지 존재 확인
    # --------------------------------------------------------

    image_name = data.get(
        "Image",
        json_path.stem + ".jpg",
    )

    image_stem = (
        Path(image_name)
        .stem
        .lower()
    )

    if image_stem not in image_stems:
        stats["MISSING_IMAGE"] += 1
        continue


    # --------------------------------------------------------
    # 대상 / 비대상 객체 분리
    # --------------------------------------------------------

    target_objects = []

    ignored_objects = []

    for obj in objects:

        code = obj.get(
            "class_name"
        )

        if code in CLASS_MAP:
            target_objects.append(
                obj
            )
        else:
            ignored_objects.append(
                obj
            )


    # ========================================================
    # 대상 객체가 하나도 없는 이미지
    # -> 학습에서 제외
    # ========================================================

    if not target_objects:

        stats["NO_TARGET"] += 1

        ignored_bbox_total += (
            len(ignored_objects)
        )

        continue


    # ========================================================
    # 대상 객체가 하나 이상 있는 이미지
    # -> 학습 사용
    # ========================================================

    stats["HAS_TARGET"] += 1


    # --------------------------------------------------------
    # 참고: TARGET_ONLY / TARGET_MIXED 구분
    # 둘 다 실제 학습에는 사용
    # --------------------------------------------------------

    if ignored_objects:

        stats[
            "TARGET_MIXED_USED"
        ] += 1

    else:

        stats[
            "TARGET_ONLY_USED"
        ] += 1


    # --------------------------------------------------------
    # 11-class 통계
    # --------------------------------------------------------

    classes_in_image = set()


    for obj in target_objects:

        code = obj[
            "class_name"
        ]

        class_name = (
            CLASS_MAP[code]
        )

        class_bbox_count[
            class_name
        ] += 1

        classes_in_image.add(
            class_name
        )

        target_bbox_total += 1


    # 이미지 수는 같은 class가 여러 bbox여도 +1
    for class_name in classes_in_image:

        class_image_count[
            class_name
        ] += 1


    # 비대상 bbox는 이후 YOLO 변환 때 무시
    ignored_bbox_total += (
        len(ignored_objects)
    )


# ============================================================
# 결과
# ============================================================

print()
print("=" * 75)
print("11-CLASS HAS_TARGET DATA COUNT")
print("=" * 75)

print(
    f"학습 사용 이미지 HAS_TARGET      : "
    f"{stats['HAS_TARGET']:,}"
)

print(
    f"  ├ TARGET_ONLY                  : "
    f"{stats['TARGET_ONLY_USED']:,}"
)

print(
    f"  └ TARGET_MIXED                 : "
    f"{stats['TARGET_MIXED_USED']:,}"
)

print()

print(
    f"학습 제외 NO_TARGET             : "
    f"{stats['NO_TARGET']:,}"
)

print(
    f"이미지 누락                     : "
    f"{stats['MISSING_IMAGE']:,}"
)

print(
    f"NO_OBJECT                       : "
    f"{stats['NO_OBJECT']:,}"
)

print(
    f"JSON_ERROR                      : "
    f"{stats['JSON_ERROR']:,}"
)


print()
print("=" * 75)
print("BBOX COUNT")
print("=" * 75)

print(
    f"학습에 사용할 대상 bbox         : "
    f"{target_bbox_total:,}"
)

print(
    f"무시할 비대상 bbox              : "
    f"{ignored_bbox_total:,}"
)


print()
print("=" * 75)
print("11-CLASS DISTRIBUTION")
print("=" * 75)


for class_name in CLASS_NAMES:

    print(
        f"{class_name:25} "
        f"images="
        f"{class_image_count[class_name]:7,} "
        f"bbox="
        f"{class_bbox_count[class_name]:7,}"
    )


print()
print("=" * 75)
print("완료 - 파일 변경 없음")
print("=" * 75)