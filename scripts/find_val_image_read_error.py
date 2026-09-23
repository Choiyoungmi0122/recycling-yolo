from pathlib import Path
import json

from PIL import Image, ImageOps


# ============================================================
# PATH
# ============================================================

IMAGE_ROOT = Path(
    "E:/recycling_data/raw/images/val/application_C"
)

JSON_ROOT = Path(
    "E:/recycling_data/raw/labels/val"
)


# ============================================================
# 우리가 사용하는 18개 AI Hub code
# ============================================================

TARGET_CODES = {
    # CAN
    "c_3",
    "c_3_01",

    # GLASS
    "c_4_01_02",
    "c_4_02_01_02",
    "c_4_02_02_02",
    "c_4_02_03_02",
    "c_4_03",
    "c_4_03_01",
    "c_4_01_01",
    "c_4_02_01_01",
    "c_4_02_02_01",
    "c_4_02_03_01",

    # PET
    "c_5_02",
    "c_5_02_01",
    "c_5_01",
    "c_5_01_01",

    # PLASTIC
    "c_6",
    "c_6_01",
}


# ============================================================
# IMAGE INDEX
# ============================================================

def build_image_index():

    image_index = {}

    for path in IMAGE_ROOT.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower()
            in {".jpg", ".jpeg", ".png"}
        ):
            image_index[path.stem.lower()] = path

    return image_index


# ============================================================
# prepare_has_target_val.py와 동일한 방식으로 읽기
# ============================================================

def test_image_read(image_path):

    with Image.open(image_path) as image:

        oriented = ImageOps.exif_transpose(image)

        width, height = oriented.size

        # 실제 픽셀까지 읽어봄
        oriented.load()

        return width, height


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("FIND VALIDATION IMAGE READ ERROR")
    print("=" * 78)

    image_index = build_image_index()

    print(f"Validation images: {len(image_index):,}")

    json_files = list(
        JSON_ROOT.rglob("*.json")
    )

    print(f"Validation JSON  : {len(json_files):,}")
    print()

    has_target = 0
    read_success = 0
    failures = []

    for index, json_path in enumerate(
        json_files,
        start=1
    ):

        if index % 5000 == 0:
            print(
                f"Checking "
                f"{index:,} / {len(json_files):,}"
            )

        try:
            with open(
                json_path,
                "r",
                encoding="utf-8-sig"
            ) as f:
                data = json.load(f)

        except Exception:
            continue

        objects = data.get(
            "objects",
            []
        )

        target_objects = [
            obj
            for obj in objects
            if obj.get("class_name")
            in TARGET_CODES
        ]

        if not target_objects:
            continue

        has_target += 1

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

        if image_path is None:

            failures.append({
                "type": "MISSING_IMAGE",
                "json": json_path,
                "image": image_name,
                "path": None,
                "codes": [
                    obj.get("class_name")
                    for obj in target_objects
                ],
                "error": "image not found"
            })

            continue

        try:

            width, height = test_image_read(
                image_path
            )

            read_success += 1

        except Exception as e:

            failures.append({
                "type": "IMAGE_READ_ERROR",
                "json": json_path,
                "image": image_name,
                "path": image_path,
                "codes": [
                    obj.get("class_name")
                    for obj in target_objects
                ],
                "error": repr(e)
            })


    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)

    print(
        f"HAS_TARGET   : {has_target:,}"
    )

    print(
        f"READ_SUCCESS : {read_success:,}"
    )

    print(
        f"FAILURES     : {len(failures):,}"
    )


    for number, failure in enumerate(
        failures,
        start=1
    ):

        print()
        print("-" * 78)

        print(
            f"[FAIL #{number}]"
        )

        print(
            "TYPE       :",
            failure["type"]
        )

        print(
            "IMAGE      :",
            failure["image"]
        )

        print(
            "IMAGE PATH :",
            failure["path"]
        )

        print(
            "JSON PATH  :",
            failure["json"]
        )

        print(
            "TARGET CODE:",
            failure["codes"]
        )

        if failure["path"] is not None:

            try:
                print(
                    "FILE SIZE  :",
                    f"{failure['path'].stat().st_size:,}",
                    "bytes"
                )

            except Exception:
                pass

        print(
            "ERROR      :",
            failure["error"]
        )


    print()
    print("=" * 78)
    print("READ-ONLY 검사 완료")
    print("=" * 78)


if __name__ == "__main__":
    main()