from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont, ImageOps


# ============================================================
# 설정
# ============================================================

DATA_ROOT = Path("E:/recycling_data")

TRAIN_LIST = DATA_ROOT / "train_has_target.txt"
VAL_LIST = DATA_ROOT / "val_has_target.txt"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "reports" / "bbox_review"

SAMPLES_PER_SPLIT = 20
RANDOM_SEED = 42


CLASS_NAMES = {
    0: "can_normal",
    1: "can_dirty",
    2: "glass_normal",
    3: "glass_dirty",
    4: "glass_packaging",
    5: "pet_normal",
    6: "pet_dirty",
    7: "pet_packaging",
    8: "pet_dirty_packaging",
    9: "plastic_normal",
    10: "plastic_dirty",
}


# ============================================================
# 이미지 경로 → YOLO label 경로
# ============================================================

def image_to_label_path(image_path: Path) -> Path:

    # 예:
    # raw/images/train/application_C/C_1/xxx.jpg
    # ↓
    # raw/labels/train/application_C/C_1/xxx.txt

    parts = list(image_path.parts)

    try:
        images_index = parts.index("images")
    except ValueError:
        raise ValueError(
            f"'images' 폴더를 경로에서 찾을 수 없음: {image_path}"
        )

    parts[images_index] = "labels"

    label_path = Path(*parts).with_suffix(".txt")

    return label_path


# ============================================================
# 폰트
# ============================================================

def get_font(size=28):

    font_candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "C:/Windows/Fonts/malgun.ttf",
    ]

    for font_path in font_candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(
                font_path,
                size=size
            )

    return ImageFont.load_default()


# ============================================================
# YOLO bbox 그리기
# ============================================================

def draw_yolo_boxes(
    image_path: Path,
    label_path: Path,
    output_path: Path,
):

    # prepare 코드와 동일하게 EXIF 방향 반영
    with Image.open(image_path) as img:

        image = ImageOps.exif_transpose(img).convert("RGB")

    width, height = image.size

    draw = ImageDraw.Draw(image)

    font = get_font(
        max(18, int(min(width, height) * 0.025))
    )

    if not label_path.exists():
        raise FileNotFoundError(
            f"YOLO label 없음: {label_path}"
        )

    lines = label_path.read_text(
        encoding="utf-8"
    ).strip().splitlines()

    box_count = 0

    for line_number, line in enumerate(
        lines,
        start=1
    ):

        if not line.strip():
            continue

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"잘못된 YOLO label 형식\n"
                f"{label_path}\n"
                f"line {line_number}: {line}"
            )

        class_id = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        # normalized → pixel
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        x1 = x_center - box_width / 2
        y1 = y_center - box_height / 2
        x2 = x_center + box_width / 2
        y2 = y_center + box_height / 2

        class_name = CLASS_NAMES.get(
            class_id,
            f"class_{class_id}"
        )

        label_text = (
            f"{class_id}: {class_name}"
        )

        # bbox
        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=max(
                3,
                int(min(width, height) * 0.004)
            ),
        )

        # 텍스트 크기
        text_bbox = draw.textbbox(
            (x1, y1),
            label_text,
            font=font,
        )

        text_width = (
            text_bbox[2] - text_bbox[0]
        )

        text_height = (
            text_bbox[3] - text_bbox[1]
        )

        text_y = max(
            0,
            y1 - text_height - 8
        )

        # 텍스트 배경
        draw.rectangle(
            [
                x1,
                text_y,
                x1 + text_width + 10,
                text_y + text_height + 8,
            ],
            fill="black",
        )

        draw.text(
            (
                x1 + 5,
                text_y + 3
            ),
            label_text,
            fill="white",
            font=font,
        )

        box_count += 1

    # 이미지 상단 정보
    info_text = (
        f"{image_path.name} | "
        f"{width}x{height} | "
        f"bbox={box_count}"
    )

    info_bbox = draw.textbbox(
        (10, 10),
        info_text,
        font=font,
    )

    draw.rectangle(
        [
            5,
            5,
            info_bbox[2] + 15,
            info_bbox[3] + 15,
        ],
        fill="black",
    )

    draw.text(
        (10, 10),
        info_text,
        fill="white",
        font=font,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    image.save(
        output_path,
        quality=92
    )

    return box_count


# ============================================================
# split 검수
# ============================================================

def review_split(
    split_name,
    list_path,
):

    print()
    print("=" * 78)
    print(
        f"{split_name.upper()} BBOX REVIEW"
    )
    print("=" * 78)

    image_paths = [
        Path(line.strip())
        for line in list_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    print(
        f"사용 이미지 목록: "
        f"{len(image_paths):,}"
    )

    random.seed(
        RANDOM_SEED
        + (0 if split_name == "train" else 1)
    )

    sample_count = min(
        SAMPLES_PER_SPLIT,
        len(image_paths)
    )

    selected = random.sample(
        image_paths,
        sample_count
    )

    output_dir = (
        OUTPUT_ROOT / split_name
    )

    success = 0
    errors = []

    for index, image_path in enumerate(
        selected,
        start=1
    ):

        try:

            label_path = image_to_label_path(
                image_path
            )

            output_path = (
                output_dir
                / f"{index:02d}_{image_path.stem}.jpg"
            )

            box_count = draw_yolo_boxes(
                image_path,
                label_path,
                output_path,
            )

            success += 1

            print(
                f"[{index:02d}/{sample_count}] "
                f"OK  "
                f"{image_path.name} "
                f"(bbox={box_count})"
            )

        except Exception as e:

            errors.append(
                (
                    image_path,
                    repr(e)
                )
            )

            print(
                f"[{index:02d}/{sample_count}] "
                f"ERROR "
                f"{image_path.name}: "
                f"{repr(e)}"
            )

    print()
    print(
        f"성공: {success}/{sample_count}"
    )

    print(
        f"오류: {len(errors)}"
    )

    print(
        f"결과 폴더: {output_dir}"
    )

    return errors


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "YOLO BBOX VISUAL VALIDATION"
    )
    print("=" * 78)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    train_errors = review_split(
        "train",
        TRAIN_LIST
    )

    val_errors = review_split(
        "val",
        VAL_LIST
    )

    print()
    print("=" * 78)
    print("FINAL")
    print("=" * 78)

    print(
        f"Train errors: "
        f"{len(train_errors)}"
    )

    print(
        f"Val errors  : "
        f"{len(val_errors)}"
    )

    print()
    print(
        "검수 이미지 저장 위치:"
    )

    print(
        OUTPUT_ROOT
    )

    print()
    print(
        "원본 이미지/YOLO label은 수정하지 않았습니다."
    )


if __name__ == "__main__":
    main()
