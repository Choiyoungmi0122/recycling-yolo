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

# 원본 Training 이미지 (TS)
IMAGE_ROOT = (
    DATA_ROOT
    / "raw"
    / "images"
    / "train"
    / "application_C"
)

# 원본 Training JSON (TL)
JSON_ROOT = (
    DATA_ROOT
    / "raw"
    / "labels"
    / "train"
)

# 새로 생성할 YOLO txt
#
# 이미지:
# E:/recycling_data/raw/images/train/application_C/C_1/xxx.jpg
#
# 라벨:
# E:/recycling_data/raw/labels/train/application_C/C_1/xxx.txt
#
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
    / "train_has_target.txt"
)

# 분석 결과는 프로젝트 reports에 저장
REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "has_target_train"
)

REPORT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. 설정
# ============================================================

# 기존에 생성된 78,688개 11-class YOLO txt를 지우고
# HAS_TARGET 기준으로 다시 생성
RESET_YOLO_LABELS = True


# ============================================================
# 3. AI Hub code -> 최종 YOLO 11-class
# ============================================================

CLASS_MAP = {
    # --------------------------------------------------------
    # CAN
    # --------------------------------------------------------
    "c_3": (0, "can_normal"),
    "c_3_01": (1, "can_dirty"),

    # --------------------------------------------------------
    # GLASS - NORMAL
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # PET
    # --------------------------------------------------------
    "c_5_02": (5, "pet_normal"),

    "c_5_02_01": (6, "pet_dirty"),

    "c_5_01": (7, "pet_packaging"),

    "c_5_01_01": (
        8,
        "pet_dirty_packaging"
    ),

    # --------------------------------------------------------
    # PLASTIC
    # --------------------------------------------------------
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
# 4. 이미지 index
# ============================================================

def build_image_index():

    print("Training 이미지 index 생성 중...")

    image_index = {}

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
    }

    for image_path in IMAGE_ROOT.rglob("*"):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in extensions:
            continue

        image_index[
            image_path.stem.lower()
        ] = image_path

    print(
        f"Training images: "
        f"{len(image_index):,}"
    )

    return image_index


# ============================================================
# 5. 이미지 좌표계 크기
# ============================================================

def get_oriented_image_size(image_path):
    """
    스마트폰 이미지의 EXIF orientation을 적용했을 때의
    실제 표시 width / height를 반환.

    기존 샘플 확인 결과 AI Hub bbox는 화면에 보이는
    방향의 좌표계와 대응하므로 bbox 자체는 재회전하지 않고,
    정규화용 width/height만 oriented size를 사용한다.
    """

    with Image.open(image_path) as image:

        oriented = ImageOps.exif_transpose(
            image
        )

        return oriented.size


# ============================================================
# 6. AI Hub bbox -> YOLO bbox
# ============================================================

def convert_bbox(
    coord,
    image_width,
    image_height,
):
    """
    AI Hub:
        x, y, width, height
        (pixel absolute coordinate)

    YOLO:
        x_center, y_center, width, height
        (0~1 normalized)
    """

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

    # --------------------------------------------------------
    # 이미지 범위 내 clip
    # --------------------------------------------------------

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

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return None

    bbox_width = (
        x2 - x1
    )

    bbox_height = (
        y2 - y1
    )

    center_x = (
        x1 + x2
    ) / 2

    center_y = (
        y1 + y2
    ) / 2

    # --------------------------------------------------------
    # 0~1 normalized
    # --------------------------------------------------------

    yolo_x = (
        center_x
        / image_width
    )

    yolo_y = (
        center_y
        / image_height
    )

    yolo_w = (
        bbox_width
        / image_width
    )

    yolo_h = (
        bbox_height
        / image_height
    )

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
# 7. YOLO txt 경로
# ============================================================

def get_yolo_label_path(
    image_path
):

    relative_path = (
        image_path.relative_to(
            IMAGE_ROOT
        )
    )

    return (
        YOLO_LABEL_ROOT
        / relative_path
    ).with_suffix(".txt")


# ============================================================
# 8. 기존 생성 YOLO label 초기화
# ============================================================

def reset_yolo_output():

    if not RESET_YOLO_LABELS:
        return

    if YOLO_LABEL_ROOT.exists():

        print()
        print(
            "기존 YOLO label 폴더 삭제:"
        )

        print(
            YOLO_LABEL_ROOT
        )

        # application_C 하위의 생성 txt만 들어 있는 폴더
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
        "HAS_TARGET TRAIN PREPROCESSING - 11 CLASS"
    )
    print("=" * 78)

    print(
        "IMAGE_ROOT:",
        IMAGE_ROOT
    )

    print(
        "JSON_ROOT :",
        JSON_ROOT
    )

    print(
        "YOLO_ROOT :",
        YOLO_LABEL_ROOT
    )

    print()

    # --------------------------------------------------------
    # 경로 확인
    # --------------------------------------------------------

    if not IMAGE_ROOT.exists():

        raise FileNotFoundError(
            f"IMAGE_ROOT 없음:\n"
            f"{IMAGE_ROOT}"
        )

    if not JSON_ROOT.exists():

        raise FileNotFoundError(
            f"JSON_ROOT 없음:\n"
            f"{JSON_ROOT}"
        )

    # --------------------------------------------------------
    # 기존 YOLO txt 초기화
    # --------------------------------------------------------

    reset_yolo_output()

    # --------------------------------------------------------
    # 이미지 index
    # --------------------------------------------------------

    image_index = (
        build_image_index()
    )

    # --------------------------------------------------------
    # JSON 목록
    # --------------------------------------------------------

    json_files = list(
        JSON_ROOT.rglob("*.json")
    )

    print(
        f"Training JSON: "
        f"{len(json_files):,}"
    )

    print()

    # --------------------------------------------------------
    # 통계
    # --------------------------------------------------------

    stats = Counter()

    class_image_count = Counter()
    class_bbox_count = Counter()

    selected_image_paths = []

    manifest_rows = []

    ignored_code_count = Counter()

    # ========================================================
    # JSON 순회
    # ========================================================

    for index, json_path in enumerate(
        json_files,
        start=1,
    ):

        if index % 10000 == 0:

            print(
                f"Processing "
                f"{index:,} / "
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
            ) as file:

                data = json.load(
                    file
                )

        except Exception as error:

            stats[
                "JSON_ERROR"
            ] += 1

            continue

        objects = data.get(
            "objects",
            [],
        )

        if not objects:

            stats[
                "NO_OBJECT"
            ] += 1

            continue

        # ----------------------------------------------------
        # 대상 / 비대상 객체 분리
        # ----------------------------------------------------

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

                    ignored_code_count[
                        code
                    ] += 1

        # ====================================================
        # 대상 객체가 하나도 없으면 이미지 제외
        # ====================================================

        if not target_objects:

            stats[
                "NO_TARGET"
            ] += 1

            stats[
                "NO_TARGET_BBOX"
            ] += len(
                ignored_objects
            )

            continue

        # ====================================================
        # 대상 객체 존재 -> 이미지 학습 사용
        # ====================================================

        stats[
            "HAS_TARGET"
        ] += 1

        if ignored_objects:

            stats[
                "TARGET_MIXED"
            ] += 1

        else:

            stats[
                "TARGET_ONLY"
            ] += 1

        stats[
            "USED_TARGET_BBOX"
        ] += len(
            target_objects
        )

        stats[
            "IGNORED_BBOX_IN_USED_IMAGE"
        ] += len(
            ignored_objects
        )

        # ----------------------------------------------------
        # 대응 이미지 찾기
        # ----------------------------------------------------

        image_name = data.get(
            "Image",
            json_path.stem + ".jpg",
        )

        image_stem = (
            Path(image_name)
            .stem
            .lower()
        )

        image_path = (
            image_index.get(
                image_stem
            )
        )

        if image_path is None:

            stats[
                "MISSING_IMAGE"
            ] += 1

            continue

        # ----------------------------------------------------
        # 이미지 크기
        # ----------------------------------------------------

        try:

            (
                image_width,
                image_height,
            ) = get_oriented_image_size(
                image_path
            )

        except Exception:

            stats[
                "IMAGE_READ_ERROR"
            ] += 1

            continue

        # ----------------------------------------------------
        # 대상 bbox만 YOLO 변환
        # ----------------------------------------------------

        yolo_lines = []

        bbox_class_names = []

        image_class_names = set()

        invalid_bbox = False

        for (
            object_index,
            obj,
        ) in target_objects:

            code = obj[
                "class_name"
            ]

            (
                class_id,
                class_name,
            ) = CLASS_MAP[
                code
            ]

            coord = (
                obj
                .get(
                    "annotation",
                    {}
                )
                .get(
                    "coord",
                    {}
                )
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

        # ----------------------------------------------------
        # 대상 bbox 중 하나라도 이상하면
        # 그 이미지 전체 제외
        # ----------------------------------------------------

        if invalid_bbox:

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
            "\n".join(
                yolo_lines
            )
            + "\n",
            encoding="utf-8",
        )

        # ----------------------------------------------------
        # train.txt용 절대경로
        # ----------------------------------------------------

        selected_image_paths.append(
            image_path
            .resolve()
            .as_posix()
        )

        # ----------------------------------------------------
        # class 통계
        # ----------------------------------------------------

        for class_name in (
            bbox_class_names
        ):

            class_bbox_count[
                class_name
            ] += 1

        for class_name in (
            image_class_names
        ):

            class_image_count[
                class_name
            ] += 1

        stats[
            "USABLE_IMAGE"
        ] += 1

        # ----------------------------------------------------
        # manifest
        # ----------------------------------------------------

        manifest_rows.append(
            {
                "image":
                    image_name,

                "image_path":
                    str(
                        image_path
                    ),

                "source_json":
                    str(
                        json_path
                    ),

                "yolo_label":
                    str(
                        label_path
                    ),

                "target_bbox_count":
                    len(
                        target_objects
                    ),

                "ignored_bbox_count":
                    len(
                        ignored_objects
                    ),

                "yolo_classes":
                    "|".join(
                        sorted(
                            image_class_names
                        )
                    ),

                "width":
                    image_width,

                "height":
                    image_height,
            }
        )

    # ========================================================
    # 10. train_has_target.txt
    # ========================================================

    TRAIN_LIST.write_text(
        "\n".join(
            selected_image_paths
        )
        + "\n",
        encoding="utf-8",
    )

    # ========================================================
    # 11. manifest CSV
    # ========================================================

    manifest_path = (
        REPORT_ROOT
        / "has_target_train_manifest.csv"
    )

    if manifest_rows:

        with open(
            manifest_path,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    manifest_rows[0].keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                manifest_rows
            )

    # ========================================================
    # 12. class distribution CSV
    # ========================================================

    distribution_path = (
        REPORT_ROOT
        / "has_target_class_distribution.csv"
    )

    distribution_rows = []

    for (
        class_id,
        class_name,
    ) in enumerate(
        CLASS_NAMES
    ):

        distribution_rows.append(
            {
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
            }
        )

    with open(
        distribution_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class_id",
                "class_name",
                "image_count",
                "bbox_count",
            ],
        )

        writer.writeheader()

        writer.writerows(
            distribution_rows
        )

    # ========================================================
    # 13. ignored code CSV
    # ========================================================

    ignored_path = (
        REPORT_ROOT
        / "ignored_code_distribution.csv"
    )

    with open(
        ignored_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "class_code",
                "bbox_count",
            ]
        )

        for (
            code,
            count,
        ) in ignored_code_count.most_common():

            writer.writerow(
                [
                    code,
                    count,
                ]
            )

    # ========================================================
    # 14. 결과
    # ========================================================

    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)

    print(
        f"전체 Training JSON            : "
        f"{len(json_files):,}"
    )

    print(
        f"HAS_TARGET                    : "
        f"{stats['HAS_TARGET']:,}"
    )

    print(
        f"  ├ TARGET_ONLY               : "
        f"{stats['TARGET_ONLY']:,}"
    )

    print(
        f"  └ TARGET_MIXED              : "
        f"{stats['TARGET_MIXED']:,}"
    )

    print(
        f"NO_TARGET                     : "
        f"{stats['NO_TARGET']:,}"
    )

    print()

    print(
        f"사용 가능한 최종 이미지      : "
        f"{stats['USABLE_IMAGE']:,}"
    )

    print(
        f"사용 대상 bbox                : "
        f"{sum(class_bbox_count.values()):,}"
    )

    print(
        f"사용 이미지 내부 무시 bbox    : "
        f"{stats['IGNORED_BBOX_IN_USED_IMAGE']:,}"
    )

    print(
        f"제외 이미지 bbox              : "
        f"{stats['NO_TARGET_BBOX']:,}"
    )

    print(
        f"이미지 누락                   : "
        f"{stats['MISSING_IMAGE']:,}"
    )

    print(
        f"이미지 읽기 오류              : "
        f"{stats['IMAGE_READ_ERROR']:,}"
    )

    print(
        f"bbox 오류 이미지              : "
        f"{stats['INVALID_BBOX_IMAGE']:,}"
    )

    print(
        f"JSON 오류                     : "
        f"{stats['JSON_ERROR']:,}"
    )

    print()

    print("=" * 78)
    print("11-CLASS DISTRIBUTION")
    print("=" * 78)

    for (
        class_id,
        class_name,
    ) in enumerate(
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

    print("=" * 78)
    print("OUTPUT")
    print("=" * 78)

    print(
        "YOLO labels :",
        YOLO_LABEL_ROOT
    )

    print(
        "Train list  :",
        TRAIN_LIST
    )

    print(
        "Manifest    :",
        manifest_path
    )

    print(
        "Class stats :",
        distribution_path
    )

    print(
        "Ignored code:",
        ignored_path
    )


if __name__ == "__main__":
    main()
