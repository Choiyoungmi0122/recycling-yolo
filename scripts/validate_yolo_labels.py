from pathlib import Path
from collections import Counter


IMAGE_ROOT = Path(
    "E:/recycling_data/raw/images/train/application_C"
)

YOLO_LABEL_ROOT = Path(
    "E:/recycling_data/raw/labels/train/application_C"
)


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


image_files = list(
    IMAGE_ROOT.rglob("*.jpg")
)

label_files = list(
    YOLO_LABEL_ROOT.rglob("*.txt")
)


print("=" * 70)
print("YOLO LABEL STRUCTURE CHECK")
print("=" * 70)

print(f"전체 JPG       : {len(image_files):,}")
print(f"YOLO TXT       : {len(label_files):,}")


image_stems = {
    p.stem.lower()
    for p in image_files
}

label_stems = {
    p.stem.lower()
    for p in label_files
}


missing_image = (
    label_stems - image_stems
)

print()
print(
    "TXT는 있는데 JPG 없음:",
    len(missing_image)
)


invalid_lines = []
class_bbox_count = Counter()

total_bbox = 0


for txt_path in label_files:

    lines = txt_path.read_text(
        encoding="utf-8"
    ).strip().splitlines()

    if not lines:
        invalid_lines.append(
            (
                txt_path,
                "EMPTY_LABEL",
                ""
            )
        )
        continue

    for line_no, line in enumerate(
        lines,
        start=1,
    ):

        parts = line.split()

        if len(parts) != 5:

            invalid_lines.append(
                (
                    txt_path,
                    f"LINE_{line_no}",
                    line
                )
            )
            continue

        try:
            class_id = int(parts[0])

            x, y, w, h = map(
                float,
                parts[1:]
            )

        except ValueError:

            invalid_lines.append(
                (
                    txt_path,
                    f"PARSE_{line_no}",
                    line
                )
            )
            continue

        if class_id not in CLASS_NAMES:

            invalid_lines.append(
                (
                    txt_path,
                    f"BAD_CLASS_{line_no}",
                    line
                )
            )
            continue

        if not (
            0 <= x <= 1
            and 0 <= y <= 1
            and 0 < w <= 1
            and 0 < h <= 1
        ):

            invalid_lines.append(
                (
                    txt_path,
                    f"BAD_BBOX_{line_no}",
                    line
                )
            )
            continue

        class_bbox_count[class_id] += 1

        total_bbox += 1


print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"정상 TXT 수 후보 : "
    f"{len(label_files):,}"
)

print(
    f"전체 bbox        : "
    f"{total_bbox:,}"
)

print(
    f"오류 line        : "
    f"{len(invalid_lines):,}"
)


print()
print("CLASS BBOX COUNT")

for class_id in range(11):

    print(
        f"{class_id:2} "
        f"{CLASS_NAMES[class_id]:25} "
        f"{class_bbox_count[class_id]:7,}"
    )


if invalid_lines:

    print()
    print("[오류 예시]")

    for item in invalid_lines[:20]:

        print(item)