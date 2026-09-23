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

# YOLO txt는 이미지와 동일한 상대구조를 갖도록 생성
YOLO_LABEL_ROOT = (
    DATA_ROOT
    / "raw"
    / "labels"
    / "train"
    / "application_C"
)

TRAIN_LIST = (
    DATA_ROOT
    / "raw"
    / "target_only_train.txt"
)

REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "target_only"
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
# IMAGE INDEX
# ============================================================

def build_image_index():

    print("Training 이미지 index 생성 중...")

    image_index = {}

    for p in IMAGE_ROOT.rglob("*.jpg"):
        image_index[p.stem.lower()] = p

    print(
        f"이미지 수: {len(image_index):,}"
    )

    return image_index


# ============================================================
# 좌표계 크기 계산
# ============================================================

def get_annotation_size(data, image_path):
    """
    AI Hub annotation 좌표는 이미 ROTATE/EXIF가 반영된
    화면 좌표계를 기준으로 저장되어 있으므로 bbox 자체는
    회전하지 않는다.

    여기서는 annotation 좌표계의 width/height만 얻는다.
    """

    info = data.get("Info", {})

    resolution = str(
        info.get("RESOLUTION", "")
    )

    rotate = info.get("ROTATE", 1)

    # --------------------------------------------------------
    # JSON metadata 우선
    # 예: "4032/3024"
    # --------------------------------------------------------

    try:
        raw_w, raw_h = [
            int(float(v))
            for v in resolution.split("/")
        ]

        rotate = int(rotate)

        # EXIF orientation 5~8은 width/height가 서로 바뀜
        if rotate in {5, 6, 7, 8}:
            return raw_h, raw_w

        return raw_w, raw_h

    except Exception:
        pass

    # --------------------------------------------------------
    # metadata가 이상하면 실제 이미지 fallback
    # --------------------------------------------------------

    with Image.open(image_path) as img:

        oriented = ImageOps.exif_transpose(
            img
        )

        return oriented.size


# ============================================================
# bbox -> YOLO
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

    # clip
    x1 = max(
        0.0,
        min(x, img_w),
    )

    y1 = max(
        0.0,
        min(y, img_h),
    )

    x2 = max(
        0.0,
        min(x + w, img_w),
    )

    y2 = max(
        0.0,
        min(y + h, img_h),
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return None

    box_w = x2 - x1
    box_h = y2 - y1

    xc = (
        (x1 + x2)
        / 2
        / img_w
    )

    yc = (
        (y1 + y2)
        / 2
        / img_h
    )

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
# YOLO label 경로
# ============================================================

def get_yolo_label_path(image_path):

    # image:
    # raw/images/train/application_C/C_1/xxx.jpg
    #
    # label:
    # raw/labels/train/application_C/C_1/xxx.txt

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

    print("=" * 70)
    print("TARGET_ONLY TRAIN DATASET")
    print("=" * 70)

    print("IMAGE_ROOT:", IMAGE_ROOT)
    print("JSON_ROOT :", JSON_ROOT)

    image_index = build_image_index()

    json_files = list(
        JSON_ROOT.rglob("*.json")
    )

    print(
        f"JSON 수: {len(json_files):,}"
    )

    stats = Counter()

    class_bbox_count = Counter()
    class_image_count = Counter()

    selected_images = []
    manifest = []

    for idx, json_path in enumerate(
        json_files,
        start=1,
    ):

        if idx % 10000 == 0:

            print(
                f"{idx:,} / "
                f"{len(json_files):,}"
            )

        # ----------------------------------------------------
        # JSON
        # ----------------------------------------------------

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

        codes = [
            obj.get("class_name")
            for obj in objects
            if obj.get("class_name")
        ]

        # ----------------------------------------------------
        # NO TARGET
        # ----------------------------------------------------

        if not any(
            code in CLASS_MAP
            for code in codes
        ):

            stats["NO_TARGET"] += 1
            continue

        # ----------------------------------------------------
        # B = MIXED
        # ----------------------------------------------------

        if not all(
            code in CLASS_MAP
            for code in codes
        ):

            stats["TARGET_MIXED"] += 1
            continue

        # ----------------------------------------------------
        # A = TARGET ONLY
        # ----------------------------------------------------

        stats["TARGET_ONLY"] += 1

        image_name = data.get(
            "Image",
            json_path.stem + ".jpg",
        )

        stem = (
            Path(image_name)
            .stem
            .lower()
        )

        image_path = image_index.get(
            stem
        )

        if image_path is None:

            stats[
                "TARGET_ONLY_MISSING_IMAGE"
            ] += 1

            manifest.append({
                "image": image_name,
                "state": "MISSING_IMAGE",
                "codes": "|".join(codes),
            })

            continue

        # ----------------------------------------------------
        # annotation coordinate size
        # ----------------------------------------------------

        try:

            img_w, img_h = (
                get_annotation_size(
                    data,
                    image_path,
                )
            )

        except Exception:

            stats[
                "IMAGE_SIZE_ERROR"
            ] += 1

            continue

        # ----------------------------------------------------
        # bbox 변환
        # ----------------------------------------------------

        yolo_lines = []

        image_class_names = set()

        invalid = False

        for obj in objects:

            code = obj.get(
                "class_name"
            )

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
                img_h,
            )

            if converted is None:

                invalid = True
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

            class_bbox_count[
                class_name
            ] += 1

            image_class_names.add(
                class_name
            )

        if invalid:

            stats[
                "INVALID_BBOX_IMAGE"
            ] += 1

            continue

        # ----------------------------------------------------
        # YOLO txt 저장
        # ----------------------------------------------------

        label_path = (
            get_yolo_label_path(
                image_path
            )
        )

        label_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        label_path.write_text(
            "\n".join(yolo_lines)
            + "\n",
            encoding="utf-8",
        )

        # ----------------------------------------------------
        # train list
        # 절대경로로 저장
        # ----------------------------------------------------

        selected_images.append(
            image_path
            .resolve()
            .as_posix()
        )

        for class_name in (
            image_class_names
        ):

            class_image_count[
                class_name
            ] += 1

        stats[
            "TARGET_ONLY_USABLE"
        ] += 1

        manifest.append({

            "image":
                image_name,

            "state":
                "TARGET_ONLY",

            "codes":
                "|".join(codes),

            "yolo_classes":
                "|".join(
                    sorted(
                        image_class_names
                    )
                ),

            "image_path":
                str(image_path),

            "label_path":
                str(label_path),

            "annotation_width":
                img_w,

            "annotation_height":
                img_h,
        })

    # ========================================================
    # TRAIN LIST
    # ========================================================

    TRAIN_LIST.write_text(
        "\n".join(
            selected_images
        )
        + "\n",
        encoding="utf-8",
    )

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest_path = (
        REPORT_ROOT
        / "target_only_train_manifest.csv"
    )

    if manifest:

        fields = sorted(
            {
                k
                for row in manifest
                for k in row
            }
        )

        with open(
            manifest_path,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:

            writer = (
                csv.DictWriter(
                    f,
                    fieldnames=fields,
                )
            )

            writer.writeheader()
            writer.writerows(
                manifest
            )

    # ========================================================
    # RESULT
    # ========================================================

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    for key, value in (
        stats.items()
    ):

        print(
            f"{key:30}"
            f": {value:,}"
        )

    print()
    print(
        "11-class distribution"
    )

    print("-" * 70)

    for name in CLASS_NAMES:

        print(
            f"{name:25} "
            f"images="
            f"{class_image_count[name]:7,} "
            f"bbox="
            f"{class_bbox_count[name]:7,}"
        )

    print()
    print("Train list:")
    print(TRAIN_LIST)

    print()
    print("Manifest:")
    print(manifest_path)


if __name__ == "__main__":
    main()
