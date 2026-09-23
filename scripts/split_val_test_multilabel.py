from pathlib import Path
from collections import Counter
import csv

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


# ============================================================
# 설정
# ============================================================

DATA_ROOT = Path("E:/recycling_data")

SOURCE_VAL_LIST = DATA_ROOT / "val_has_target.txt"

VAL_FINAL_LIST = DATA_ROOT / "val_final.txt"
TEST_FINAL_LIST = DATA_ROOT / "test_final.txt"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "val_test_split"

REPORT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_SEED = 42
TEST_SIZE = 0.5


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

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# 이미지 경로 -> YOLO label 경로
# ============================================================

def image_to_label_path(image_path: Path) -> Path:

    parts = list(image_path.parts)

    try:
        image_index = parts.index("images")
    except ValueError:
        raise ValueError(
            f"'images'를 경로에서 찾지 못했습니다: {image_path}"
        )

    parts[image_index] = "labels"

    return Path(*parts).with_suffix(".txt")


# ============================================================
# YOLO label 읽기
# ============================================================

def read_yolo_label(label_path: Path):

    if not label_path.exists():
        raise FileNotFoundError(
            f"YOLO label 없음: {label_path}"
        )

    class_ids = []

    lines = label_path.read_text(
        encoding="utf-8"
    ).splitlines()

    for line_number, line in enumerate(
        lines,
        start=1
    ):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"잘못된 YOLO label 형식\n"
                f"{label_path}\n"
                f"line={line_number}: {line}"
            )

        class_id = int(parts[0])

        if not 0 <= class_id < NUM_CLASSES:
            raise ValueError(
                f"잘못된 class id: {class_id}\n"
                f"{label_path}"
            )

        class_ids.append(class_id)

    if not class_ids:
        raise ValueError(
            f"bbox가 없는 YOLO label: {label_path}"
        )

    return class_ids


# ============================================================
# split 통계 계산
# ============================================================

def calculate_stats(image_paths):

    image_counts = Counter()
    bbox_counts = Counter()

    total_bbox = 0

    for image_path in image_paths:

        label_path = image_to_label_path(
            image_path
        )

        class_ids = read_yolo_label(
            label_path
        )

        # 이미지 단위 클래스 수
        unique_classes = set(class_ids)

        for class_id in unique_classes:
            image_counts[class_id] += 1

        # bbox 단위 클래스 수
        for class_id in class_ids:
            bbox_counts[class_id] += 1
            total_bbox += 1

    return (
        image_counts,
        bbox_counts,
        total_bbox,
    )


# ============================================================
# 분포 CSV 저장
# ============================================================

def save_distribution_csv(
    val_paths,
    test_paths,
):

    (
        val_image_counts,
        val_bbox_counts,
        val_bbox_total,
    ) = calculate_stats(val_paths)

    (
        test_image_counts,
        test_bbox_counts,
        test_bbox_total,
    ) = calculate_stats(test_paths)

    output_path = (
        REPORT_ROOT
        / "val_test_class_distribution.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "class_id",
            "class_name",
            "val_image_count",
            "test_image_count",
            "val_bbox_count",
            "test_bbox_count",
            "val_bbox_ratio",
            "test_bbox_ratio",
        ])

        for class_id, class_name in enumerate(
            CLASS_NAMES
        ):

            val_bbox = val_bbox_counts[class_id]
            test_bbox = test_bbox_counts[class_id]

            bbox_sum = val_bbox + test_bbox

            if bbox_sum > 0:
                val_ratio = val_bbox / bbox_sum
                test_ratio = test_bbox / bbox_sum
            else:
                val_ratio = 0
                test_ratio = 0

            writer.writerow([
                class_id,
                class_name,
                val_image_counts[class_id],
                test_image_counts[class_id],
                val_bbox,
                test_bbox,
                round(val_ratio, 6),
                round(test_ratio, 6),
            ])

    return (
        output_path,
        val_bbox_total,
        test_bbox_total,
        val_image_counts,
        test_image_counts,
        val_bbox_counts,
        test_bbox_counts,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("VALIDATION -> VAL / TEST MULTI-LABEL STRATIFIED SPLIT")
    print("=" * 80)

    if not SOURCE_VAL_LIST.exists():
        raise FileNotFoundError(
            SOURCE_VAL_LIST
        )

    # --------------------------------------------------------
    # 기존 Validation 목록
    # --------------------------------------------------------

    image_paths = [
        Path(line.strip())
        for line in SOURCE_VAL_LIST.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    print(
        f"Original Validation images: "
        f"{len(image_paths):,}"
    )

    # --------------------------------------------------------
    # Multi-label matrix 생성
    # --------------------------------------------------------

    y = np.zeros(
        (
            len(image_paths),
            NUM_CLASSES
        ),
        dtype=np.int8,
    )

    print()
    print("YOLO label 분석 중...")

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        if index % 2000 == 0:
            print(
                f"Processing "
                f"{index:,} / {len(image_paths):,}"
            )

        label_path = image_to_label_path(
            image_path
        )

        class_ids = read_yolo_label(
            label_path
        )

        # 한 이미지에서 같은 클래스 bbox가 여러 개 있어도
        # stratification에서는 클래스 존재 여부만 1로 기록
        for class_id in set(class_ids):
            y[index - 1, class_id] = 1

    # --------------------------------------------------------
    # Multi-label stratified split
    # --------------------------------------------------------

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    dummy_x = np.zeros(
        (
            len(image_paths),
            1
        )
    )

    val_indices, test_indices = next(
        splitter.split(
            dummy_x,
            y
        )
    )

    val_paths = [
        image_paths[i]
        for i in val_indices
    ]

    test_paths = [
        image_paths[i]
        for i in test_indices
    ]

    # --------------------------------------------------------
    # 재현성을 위해 파일 내부 순서 정렬
    # --------------------------------------------------------

    val_paths = sorted(
        val_paths,
        key=lambda x: str(x).lower()
    )

    test_paths = sorted(
        test_paths,
        key=lambda x: str(x).lower()
    )

    # --------------------------------------------------------
    # overlap 확인
    # --------------------------------------------------------

    val_set = {
        str(p).lower()
        for p in val_paths
    }

    test_set = {
        str(p).lower()
        for p in test_paths
    }

    overlap = val_set & test_set

    if overlap:
        raise RuntimeError(
            f"Val/Test overlap 발생: "
            f"{len(overlap):,}장"
        )

    if (
        len(val_paths)
        + len(test_paths)
        != len(image_paths)
    ):
        raise RuntimeError(
            "Val + Test 이미지 수가 "
            "원본 Validation과 일치하지 않습니다."
        )

    # --------------------------------------------------------
    # txt 저장
    # --------------------------------------------------------

    VAL_FINAL_LIST.write_text(
        "\n".join(
            p.as_posix()
            for p in val_paths
        ) + "\n",
        encoding="utf-8",
    )

    TEST_FINAL_LIST.write_text(
        "\n".join(
            p.as_posix()
            for p in test_paths
        ) + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 통계
    # --------------------------------------------------------

    (
        distribution_path,
        val_bbox_total,
        test_bbox_total,
        val_image_counts,
        test_image_counts,
        val_bbox_counts,
        test_bbox_counts,
    ) = save_distribution_csv(
        val_paths,
        test_paths
    )

    # --------------------------------------------------------
    # 결과 출력
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("SPLIT RESULT")
    print("=" * 80)

    print(
        f"Original Val : {len(image_paths):,}"
    )

    print(
        f"Final Val    : {len(val_paths):,}"
    )

    print(
        f"Final Test   : {len(test_paths):,}"
    )

    print(
        f"Overlap      : {len(overlap):,}"
    )

    print()

    print(
        f"Val bbox     : {val_bbox_total:,}"
    )

    print(
        f"Test bbox    : {test_bbox_total:,}"
    )

    print(
        f"Total bbox   : "
        f"{val_bbox_total + test_bbox_total:,}"
    )

    print()
    print("=" * 80)
    print("CLASS DISTRIBUTION")
    print("=" * 80)

    print(
        f"{'ID':>2} "
        f"{'CLASS':25} "
        f"{'VAL_IMG':>8} "
        f"{'TEST_IMG':>9} "
        f"{'VAL_BOX':>8} "
        f"{'TEST_BOX':>9}"
    )

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{class_id:2d} "
            f"{class_name:25} "
            f"{val_image_counts[class_id]:8,d} "
            f"{test_image_counts[class_id]:9,d} "
            f"{val_bbox_counts[class_id]:8,d} "
            f"{test_bbox_counts[class_id]:9,d}"
        )

    print()
    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print(
        "Original Val :",
        SOURCE_VAL_LIST
    )

    print(
        "Final Val    :",
        VAL_FINAL_LIST
    )

    print(
        "Final Test   :",
        TEST_FINAL_LIST
    )

    print(
        "Distribution :",
        distribution_path
    )

    print()
    print(
        "원본 이미지와 YOLO label은 "
        "복사/수정/삭제하지 않았습니다."
    )


if __name__ == "__main__":
    main()
