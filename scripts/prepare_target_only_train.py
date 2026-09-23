from pathlib import Path
from collections import Counter
import csv
import json

from PIL import Image, ImageOps


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(
    "E:/recycling_data"
)

IMAGE_ROOT = (
    DATA_ROOT
    / "raw"
    / "images"
    / "train"
    / "application_C"
)

JSON_ROOT = (
    DATA_ROOT
    / "raw"
    / "labels"
    / "train"
)

# 새로 생성할 YOLO txt
YOLO_LABEL_ROOT = (
    DATA_ROOT
    / "raw"
    / "labels"
    / "train"
    / "application_C"
)

# 실제 학습에 사용할 이미지 경로 목록
TRAIN_LIST = (
    DATA_ROOT
    / "target_only_train.txt"
)

# 작은 분석 결과는 C: 프로젝트에 저장
REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "target_only_train"
)

REPORT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 18 AI Hub codes -> 11 YOLO classes
# ============================================================

CLASS_MAP = {

    # CAN
    "c_3": (0, "can_normal"),
    "c_3_01": (1, "can_dirty"),

    # GLASS - NORMAL
    "c_4_01_02": (2, "glass_normal"),
    "c_4_02_01_02": (2, "glass_normal"),
    "c_4_02_02_02": (2, "glass_normal"),
    "c_4_02_03_02": (2, "glass_normal"),
    "c_4_03": (2, "glass_normal"),

    # GLASS - DIRTY
    "c_4_03_01": (3, "glass_dirty"),

    # GLASS - PACKAGING
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
# IMAGE INDEX
# ============================================================

def build_image_index():

    print("Training 이미지 index 생성 중...")

    image_index = {}

    for image_path in IMAGE_ROOT.rglob("*.jpg"):
        image_index[
            image_path.stem.lower()
        ] = image_path

    print(
        f"Training images: "
        f"{len(image_index):,}"
    )

    return image_index


# ============================================================
# 이미지 표시 좌표계 크기
# ============================================================

def get_oriented_image_size(image_path):
    """
    AI Hub 스마트폰 이미지에는 EXIF orientation이 존재할 수 있음.

    bbox 자체는 AI Hub JSON의 좌표를 그대로 사용하고,
    정규화에 사용할 width/height만 실제 표시 방향 기준으로 얻는다.
    """

    with Image.open(image_path) as image:

        oriented = ImageOps.exif_transpose(
            image
        )

        return oriented.size


# ============================================================
# AI Hub bbox -> YOLO bbox
# ============================================================

def convert_bbox(coord, img_w, img_h):

    try:

        x = float(coord["x"])
        y = float(coord["y"])
        w = float(coord["width"])
        h = float(coord["height"])

    except (
        KeyError,
        TypeError,
        ValueError,
    ):

        return None

    # ------------------------------------------
    # image boundary clip
    # ------------------------------------------

    x1 = max(
        0.0,
        min(x, img_w)
    )

    y1 = max(
        0.0,
        min(y, img_h)
    )

    x2 = max(
        0.0,
        min(x + w, img_w)
    )

    y2 = max(
        0.0,
        min(y + h, img_h)
    )

    if x2 <= x1 or y2 <= y1:
        return None

    box_w = x2 - x1
    box_h = y2 - y1

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # ------------------------------------------
    # YOLO normalization: 0~1
    # ------------------------------------------

    xc = center_x / img_w
    yc = center_y / img_h

    nw = box_w / img_w
    nh = box_h / img_h

    if not (
        0 <= xc <= 1
        and 0 <= yc <= 1
        and 0 < nw <= 1
        and 0 < nh <= 1
    ):
        return None

    return xc, yc, nw, nh


# ============================================================
# YOLO txt output path
# ============================================================

def get_yolo_label_path(image_path):

    # 예:
    #
    # images/train/application_C/C_1/a.jpg
    #
    # ->
    #
    # labels/train/application_C/C_1/a.txt

    relative = image_path.relative_to(
        IMAGE_ROOT
    )

    return (
        YOLO_LABEL_ROOT
        / relative
    ).with_suffix(".txt")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("TARGET_ONLY TRAIN PREPROCESSING")
    print("=" * 75)

    print("IMAGE_ROOT :", IMAGE_ROOT)
    print("JSON_ROOT  :", JSON_ROOT)
    print()

    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(
            IMAGE_ROOT
        )

    if not JSON_ROOT.exists():
        raise FileNotFoundError(
            JSON_ROOT
        )

    image_index = build_image_index()

    json_files = list(
        JSON_ROOT.rglob("*.json")
    )

    print(
        f"Training JSON: "
        f"{len(json_files):,}"
    )

    print()

    stats = Counter()

    # 최종 11-class별
    class_bbox_count = Counter()
    class_image_count = Counter()

    selected_images = []

    manifest_rows = []

    missing_rows = []

    # ========================================================
    # JSON 순회
    # ========================================================

    for idx, json_path in enumerate(
        json_files,
        start=1,
    ):

        if idx % 10000 == 0:

            print(
                f"Processing "
                f"{idx:,} / "
                f"{len(json_files):,}"
            )

        # ----------------------------------------------------
        # JSON READ
        # ----------------------------------------------------

        try:

            with open(
                json_path,
                "r",
                encoding="utf-8-sig",
            ) as f:

                data = json.load(f)

        except Exception as e:

            stats["JSON_ERROR"] += 1

            continue

        objects = data.get(
            "objects",
            []
        )

        if not objects:

            stats["NO_OBJECT"] += 1
            continue

        codes = [
            obj.get("class_name")
            for obj in objects
            if obj.get("class_name")
        ]

        # class_name이 없는 annotation 존재
        if len(codes) != len(objects):

            stats[
                "INVALID_OBJECT_CLASS"
            ] += 1

            continue

        # ====================================================
        # C: NO TARGET
        # ====================================================

        if not any(
            code in CLASS_MAP
            for code in codes
        ):

            stats["NO_TARGET"] += 1

            continue

        # ====================================================
        # B: TARGET + OTHER CLASS
        # ====================================================

        if not all(
            code in CLASS_MAP
            for code in codes
        ):

            stats["TARGET_MIXED"] += 1

            continue

        # ====================================================
        # A: TARGET ONLY
        # ====================================================

        stats[
            "TARGET_ONLY_LABEL"
        ] += 1

        image_name = data.get(
            "Image",
            json_path.stem + ".jpg"
        )

        image_stem = (
            Path(image_name)
            .stem
            .lower()
        )

        image_path = image_index.get(
            image_stem
        )

        # ----------------------------------------------------
        # 이미지 누락
        # ----------------------------------------------------

        if image_path is None:

            stats[
                "TARGET_ONLY_MISSING_IMAGE"
            ] += 1

            missing_rows.append({

                "stem":
                    image_stem,

                "json_file":
                    str(json_path),

                "codes":
                    "|".join(codes),

                "mapped_classes":
                    "|".join(
                        sorted({
                            CLASS_MAP[c][1]
                            for c in codes
                        })
                    ),
            })

            continue

        # ----------------------------------------------------
        # annotation 기준 이미지 크기
        # ----------------------------------------------------

        try:

            img_w, img_h = (
                get_oriented_image_size(
                    image_path
                )
            )

        except Exception:

            stats[
                "IMAGE_READ_ERROR"
            ] += 1

            continue

        # ----------------------------------------------------
        # bbox 변환
        # ----------------------------------------------------

        yolo_lines = []

        image_classes = set()

        bbox_records = []

        invalid_bbox = False

        for obj in objects:

            code = obj[
                "class_name"
            ]

            class_id, class_name = (
                CLASS_MAP[code]
            )

            coord = (
                obj
                .get("annotation", {})
                .get("coord", {})
            )

            converted = convert_bbox(
                coord,
                img_w,
                img_h
            )

            if converted is None:

                invalid_bbox = True
                break

            xc, yc, nw, nh = (
                converted
            )

            yolo_lines.append(
                f"{class_id} "
                f"{xc:.6f} "
                f"{yc:.6f} "
                f"{nw:.6f} "
                f"{nh:.6f}"
            )

            bbox_records.append(
                class_name
            )

            image_classes.add(
                class_name
            )

        if invalid_bbox:

            stats[
                "INVALID_BBOX_IMAGE"
            ] += 1

            continue

        # ----------------------------------------------------
        # YOLO .txt 저장
        # ----------------------------------------------------

        label_path = (
            get_yolo_label_path(
                image_path
            )
        )

        label_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        label_path.write_text(
            "\n".join(
                yolo_lines
            ) + "\n",
            encoding="utf-8"
        )

        # ----------------------------------------------------
        # 학습 목록
        # ----------------------------------------------------

        selected_images.append(
            image_path
            .resolve()
            .as_posix()
        )

        # ----------------------------------------------------
        # 통계
        # ----------------------------------------------------

        for class_name in bbox_records:

            class_bbox_count[
                class_name
            ] += 1

        for class_name in image_classes:

            class_image_count[
                class_name
            ] += 1

        stats[
            "TARGET_ONLY_USABLE"
        ] += 1

        # ----------------------------------------------------
        # manifest
        # ----------------------------------------------------

        manifest_rows.append({

            "image":
                image_name,

            "source_json":
                str(json_path),

            "image_path":
                str(image_path),

            "yolo_label":
                str(label_path),

            "original_codes":
                "|".join(codes),

            "yolo_classes":
                "|".join(
                    sorted(
                        image_classes
                    )
                ),

            "bbox_count":
                len(yolo_lines),

            "width":
                img_w,

            "height":
                img_h,
        })

    # ========================================================
    # train.txt
    # ========================================================

    TRAIN_LIST.write_text(
        "\n".join(
            selected_images
        ) + "\n",
        encoding="utf-8"
    )

    # ========================================================
    # manifest CSV
    # ========================================================

    manifest_path = (
        REPORT_ROOT
        / "target_only_train_manifest.csv"
    )

    if manifest_rows:

        fields = list(
            manifest_rows[0].keys()
        )

        with open(
            manifest_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fields
            )

            writer.writeheader()

            writer.writerows(
                manifest_rows
            )

    # ========================================================
    # missing CSV
    # ========================================================

    missing_path = (
        REPORT_ROOT
        / "missing_target_only_images.csv"
    )

    if missing_rows:

        fields = list(
            missing_rows[0].keys()
        )

        with open(
            missing_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fields
            )

            writer.writeheader()

            writer.writerows(
                missing_rows
            )

    # ========================================================
    # class distribution CSV
    # ========================================================

    distribution_path = (
        REPORT_ROOT
        / "target_only_class_distribution.csv"
    )

    distribution_rows = []

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        distribution_rows.append({

            "class_id":
                class_id,

            "class_name":
                class_name,

            "image_count":
                class_image_count[
                    class_name
                ],

            "bbox_count":
                class_bbox_count[
                    class_name
                ],
        })

    with open(
        distribution_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_id",
                "class_name",
                "image_count",
                "bbox_count",
            ]
        )

        writer.writeheader()

        writer.writerows(
            distribution_rows
        )

    # ========================================================
    # RESULT
    # ========================================================

    print()
    print("=" * 75)
    print("RESULT")
    print("=" * 75)

    result_order = [
        "NO_OBJECT",
        "NO_TARGET",
        "TARGET_MIXED",
        "TARGET_ONLY_LABEL",
        "TARGET_ONLY_MISSING_IMAGE",
        "IMAGE_READ_ERROR",
        "INVALID_BBOX_IMAGE",
        "TARGET_ONLY_USABLE",
        "JSON_ERROR",
        "INVALID_OBJECT_CLASS",
    ]

    for key in result_order:

        print(
            f"{key:30}"
            f": {stats[key]:,}"
        )

    print()
    print("=" * 75)
    print("11-CLASS DISTRIBUTION")
    print("=" * 75)

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
    print("=" * 75)
    print("OUTPUT")
    print("=" * 75)

    print(
        "Train list :",
        TRAIN_LIST
    )

    print(
        "Manifest   :",
        manifest_path
    )

    print(
        "Class stat :",
        distribution_path
    )

    if missing_rows:

        print(
            "Missing    :",
            missing_path
        )


if __name__ == "__main__":
    main()
