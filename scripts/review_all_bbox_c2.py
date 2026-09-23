from pathlib import Path
import json

from PIL import Image, ImageDraw, ImageOps, ImageFont
FONT = ImageFont.truetype(
    "C:/Windows/Fonts/arial.ttf",
    48
)

# ============================================================
# 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABEL_ROOT = Path(
    "E:/recycling_data/raw/labels/train"
)

IMAGE_ROOT = Path(
    "E:/recycling_data/raw/images/train/application_C/C_1"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "review_samples_c2"
    / "all_bbox"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 확인할 이미지
# ============================================================

IMAGE_NAME = "C_20220816_007433.jpg"


# ============================================================
# 원본 이미지 찾기
# ============================================================

image_candidates = list(
    IMAGE_ROOT.rglob(IMAGE_NAME)
)

if not image_candidates:
    raise FileNotFoundError(
        f"이미지를 찾지 못했습니다: {IMAGE_NAME}"
    )

image_path = image_candidates[0]


# ============================================================
# 대응 JSON 찾기
# ============================================================

json_name = Path(IMAGE_NAME).with_suffix(".json").name

json_candidates = list(
    LABEL_ROOT.rglob(json_name)
)

if not json_candidates:
    raise FileNotFoundError(
        f"JSON을 찾지 못했습니다: {json_name}"
    )

json_path = json_candidates[0]


print("IMAGE:", image_path)
print("JSON :", json_path)


# ============================================================
# JSON 읽기
# ============================================================

with open(
    json_path,
    "r",
    encoding="utf-8-sig",
) as f:
    data = json.load(f)


objects = data.get("objects", [])

print(f"annotation 수: {len(objects)}")


# ============================================================
# 이미지 열기
# ============================================================

image = Image.open(image_path)
image.load()

# 기존 review_samples.py와 동일하게 EXIF 방향 처리
image = ImageOps.exif_transpose(image)
image = image.convert("RGB")

draw = ImageDraw.Draw(image)


# ============================================================
# 모든 bbox 그리기
# ============================================================

for idx, obj in enumerate(objects):

    class_name = obj.get(
        "class_name",
        "UNKNOWN",
    )

    coord = (
        obj
        .get("annotation", {})
        .get("coord", {})
    )

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
        continue

    x2 = x + w
    y2 = y + h

    # bbox
    draw.rectangle(
        [x, y, x2, y2],
        outline="red",
        width=8,
    )

    # class 이름
    text = f"{idx}: {class_name}"

    draw.text(
    (
        x,
        max(0, y - 55),
    ),
    text,
    fill="red",
    font=FONT,
    )

    print(
        f"{idx}: {class_name} "
        f"bbox=({x:.0f}, {y:.0f}, {w:.0f}, {h:.0f})"
    )


# ============================================================
# 저장
# ============================================================

# 확인하기 편한 크기로 축소
image.thumbnail(
    (1800, 1800)
)

output_path = (
    OUTPUT_DIR
    / "C_20220824_001594_all_bbox.jpg"
)

image.save(
    output_path,
    quality=95,
)

print()
print("저장 완료:")
print(output_path)
