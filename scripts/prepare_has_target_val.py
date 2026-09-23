from pathlib import Path
from collections import Counter
import csv
import json
import shutil

from PIL import Image, ImageOps


# ============================================================
# 1. PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(
    "E:/recycling_data"
)

# VS
IMAGE_ROOT = (
    DATA_ROOT
    / "raw"
    / "images"
    / "val"
    / "application_C"
)

# VL
JSON_ROOT = (
    DATA_ROOT
    / "raw"
    / "labels"
    / "val"
)

# Validation용 YOLO txt
YOLO_LABEL_ROOT = (
    DATA_ROOT
    / "raw"
    / "labels"
    / "val"
    / "application_C"
)

# Validation에 실제 사용할 이미지 목록
VAL_LIST = (
    DATA_ROOT
    / "val_has_target.txt"
)

REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "has_target_val"
)

REPORT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. 설정
# ============================================================

RESET_YOLO_LABELS = True


# ============================================================
# 3. AI Hub -> YOLO 11-class
# ============================================================

CLASS_MAP = {
    # CAN
    "c_3": (0, "can_normal"),
    "c_3_01": (1, "can_dirty"),

    # GLASS NORMAL
    "c_4_01_02": (2, "glass_normal"),
    "c_4_02_01_02": (2, "glass_normal"),
    "c_4_02_02_02": (2, "glass_normal"),
    "c_4_02_03_02": (2, "glass_normal"),
    "c_4_03": (2, "glass_normal"),

    # GLASS DIRTY
    "c_4_03_01": (3, "glass_dirty"),

    # GLASS PACKAGING
    "c_4_01_01": (4, "glass_packaging"),
    "c_4_02_01_01": (4, "glass_packaging"),
    "c_4_02_02_01": (4, "glass_packaging"),
    "c_4_02_03_01": (4, "glass_packaging"),

    # PET
    "c_5_02": (5, "pet_normal"),
    "c_5_02_01": (6, "pet_dirty"),
    "c_5_01": (7, "pet_packaging"),
    "c_5_01_01": (8, "pet_dirty_packaging"),

    # PLASTIC
    "c_6": (9, "plastic_normal"),
    "c_6_01": (10, "plastic_dirty"),
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
# 4. IMAGE INDEX
# ============================================================

def build_image_index():

    print("Validation 이미지 index 생성 중...")

    image_index = {}

    for path in IMAGE_ROOT.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
        }:
            continue

        image_index[path.stem.lower()] = path

    print(
        f"Validation images: {len(image_index):,}"
    )

    return image_index


# ============================================================
# 5. 이미지 표시 크기
# ============================================================

def get_oriented_image_size(image_path):

    with Image.open(image_path) as image:

        oriented = ImageOps.exif_transpose(
            image
        )

        return oriented.size


# ============================================================
# 6. bbox -> YOLO
# ============================================================

def convert_bbox(
    coord,
    image_width,
    image_height,
):

    try:
        x = float(coord["x"])
        y = float(coord["y"])
        width = float(coord["width"])
        height = float(coord["height"])

    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        return None

    x1 = max(
        0.0,
        min(x, image_width),
    )

    y1 = max(
        0.0,
        min(y, image_height),
    )

    x2 = max(
        0.0,
        min(
            x + width,
            image_width,
        ),
    )

    y2 = max(
        0.0,
        min(
            y + height,
            image_height,
        ),
    )

    if x2 <= x1 or y2 <= y1:
        return None

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    yolo_x = center_x / image_width
    yolo_y = center_y / image_height
    yolo_w = bbox_width / image_width
    yolo_h = bbox_height / image_height

    if not (
        0 <= yolo_x <= 1
        and 0 <= yolo_y <= 1
        and 0 < yolo_w <= 1
        and 0 < yolo_h <= 1
    ):
        return None

    return (
        yolo_x,
        yolo_y,
        yolo_w,
        yolo_h,
    )


# ============================================================
# 7. label path
# ============================================================

def get_yolo_label_path(image_path):

    relative_path = image_path.relative_to(
        IMAGE_ROOT
    )

    return (
        YOLO_LABEL_ROOT
        / relative_path
    ).with_suffix(".txt")


# ============================================================
# 8. 초기화
# ============================================================

def reset_yolo_output():

    if not RESET_YOLO_LABELS:
        return

    if YOLO_LABEL_ROOT.exists():

        print(
            "기존 Validation YOLO label 폴더 삭제:"
        )

        print(YOLO_LABEL_ROOT)

        shutil.rmtree(
            YOLO_LABEL_ROOT
        )

    YOLO_LABEL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "HAS_TARGET VALIDATION PREPROCESSING - 11 CLASS"
    )
    print("=" * 78)

    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(IMAGE_ROOT)

    if not JSON_ROOT.exists():
        raise FileNotFoundError(JSON_ROOT)

    reset_yolo_output()

    image_index = build_image_index()

    json_files = list(
        JSON_ROOT.rglob("*.json")
    )

    print(
        f"Validation JSON: {len(json_files):,}"
    )

    stats = Counter()

    class_image_count = Counter()
    class_bbox_count = Counter()

    selected_image_paths = []
    manifest_rows = []
    ignored_code_count = Counter()

    for index, json_path in enumerate(
        json_files,
        start=1,
    ):

        if index % 5000 == 0:
            print(
                f"Processing "
                f"{index:,} / "
                f"{len(json_files):,}"
            )

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
            [],
        )

        if not objects:
            stats["NO_OBJECT"] += 1
            continue

        target_objects = []
        ignored_objects = []

        for object_index, obj in enumerate(
            objects
        ):

            code = obj.get(
                "class_name"
            )

            if code in CLASS_MAP:
                target_objects.append(
                    (
                        object_index,
                        obj,
                    )
                )

            else:
                ignored_objects.append(
                    (
                        object_index,
                        obj,
                    )
                )

                if code:
                    ignored_code_count[code] += 1

        # 대상 객체 없음
        if not target_objects:

            stats["NO_TARGET"] += 1
            stats["NO_TARGET_BBOX"] += len(
                ignored_objects
            )

            continue

        stats["HAS_TARGET"] += 1

        if ignored_objects:
            stats["TARGET_MIXED"] += 1
        else:
            stats["TARGET_ONLY"] += 1

        stats[
            "IGNORED_BBOX_IN_USED_IMAGE"
        ] += len(ignored_objects)

        image_name = data.get(
            "Image",
            json_path.stem + ".jpg",
        )

        image_stem = (
            Path(image_name)
            .stem
            .lower()
        )

        image_path = image_index.get(
            image_stem
        )

        if image_path is None:
            stats["MISSING_IMAGE"] += 1
            continue

        try:

            (
                image_width,
                image_height,
            ) = get_oriented_image_size(
                image_path
            )

        except Exception:

            stats["IMAGE_READ_ERROR"] += 1
            continue

        yolo_lines = []

        bbox_class_names = []
        image_class_names = set()

        invalid_bbox = False

        for _, obj in target_objects:

            code = obj["class_name"]

            (
                class_id,
                class_name,
            ) = CLASS_MAP[code]

            coord = (
                obj
                .get("annotation", {})
                .get("coord", {})
            )

            converted = convert_bbox(
                coord,
                image_width,
                image_height,
            )

            if converted is None:

                invalid_bbox = True
                break

            (
                x,
                y,
                width,
                height,
            ) = converted

            yolo_lines.append(
                f"{class_id} "
                f"{x:.6f} "
                f"{y:.6f} "
                f"{width:.6f} "
                f"{height:.6f}"
            )

            bbox_class_names.append(
                class_name
            )

            image_class_names.add(
                class_name
            )

        if invalid_bbox:

            stats["INVALID_BBOX_IMAGE"] += 1
            continue

        label_path = get_yolo_label_path(
            image_path
        )

        label_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        label_path.write_text(
            "\n".join(yolo_lines) + "\n",
            encoding="utf-8",
        )

        selected_image_paths.append(
            image_path.resolve().as_posix()
        )

        for class_name in bbox_class_names:
            class_bbox_count[class_name] += 1

        for class_name in image_class_names:
            class_image_count[class_name] += 1

        stats["USABLE_IMAGE"] += 1

        manifest_rows.append({
            "image": image_name,
            "image_path": str(image_path),
            "source_json": str(json_path),
            "yolo_label": str(label_path),
            "target_bbox_count": len(target_objects),
            "ignored_bbox_count": len(ignored_objects),
            "yolo_classes": "|".join(
                sorted(image_class_names)
            ),
            "width": image_width,
            "height": image_height,
        })

    # ========================================================
    # val_has_target.txt
    # ========================================================

    VAL_LIST.write_text(
        "\n".join(
            selected_image_paths
        ) + "\n",
        encoding="utf-8",
    )

    # ========================================================
    # manifest
    # ========================================================

    manifest_path = (
        REPORT_ROOT
        / "has_target_val_manifest.csv"
    )

    if manifest_rows:

        with open(
            manifest_path,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    manifest_rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                manifest_rows
            )

    # ========================================================
    # distribution
    # ========================================================

    distribution_path = (
        REPORT_ROOT
        / "has_target_val_class_distribution.csv"
    )

    with open(
        distribution_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "class_id",
            "class_name",
            "image_count",
            "bbox_count",
        ])

        for class_id, class_name in enumerate(
            CLASS_NAMES
        ):

            writer.writerow([
                class_id,
                class_name,
                class_image_count[class_name],
                class_bbox_count[class_name],
            ])

    # ========================================================
    # 결과
    # ========================================================

    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)

    print(
        f"HAS_TARGET       : "
        f"{stats['HAS_TARGET']:,}"
    )

    print(
        f"TARGET_ONLY      : "
        f"{stats['TARGET_ONLY']:,}"
    )

    print(
        f"TARGET_MIXED     : "
        f"{stats['TARGET_MIXED']:,}"
    )

    print(
        f"NO_TARGET        : "
        f"{stats['NO_TARGET']:,}"
    )

    print()

    print(
        f"USABLE_IMAGE     : "
        f"{stats['USABLE_IMAGE']:,}"
    )

    print(
        f"MISSING_IMAGE    : "
        f"{stats['MISSING_IMAGE']:,}"
    )

    print(
        f"IMAGE_READ_ERROR : "
        f"{stats['IMAGE_READ_ERROR']:,}"
    )

    print(
        f"INVALID_BBOX     : "
        f"{stats['INVALID_BBOX_IMAGE']:,}"
    )

    print()

    print("=" * 78)
    print("11-CLASS DISTRIBUTION")
    print("=" * 78)

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{class_id:2} "
            f"{class_name:25} "
            f"images="
            f"{class_image_count[class_name]:7,} "
            f"bbox="
            f"{class_bbox_count[class_name]:7,}"
        )

    print()

    print("Val list :", VAL_LIST)
    print("Manifest :", manifest_path)


if __name__ == "__main__":
    main()
