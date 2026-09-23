from pathlib import Path
import json
import random

from PIL import Image, ImageDraw, ImageOps, ImageFont


# ============================================================
# 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Training JSON
LABEL_ROOT = Path(
    "E:/recycling_data/raw/labels/train"
)

# C1 + C2 이미지
IMAGE_ROOTS = [
    Path("E:/recycling_data/raw/images/train/application_C/C_1"),
    Path("E:/recycling_data/raw/images/train/application_C/C_2"),
]

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "abnormal_review_c1_c2"
)


# ============================================================
# 확인할 클래스
# ============================================================

REVIEW_CLASSES = {

    # --------------------------------------------------------
    # 이물질
    # --------------------------------------------------------

    "dirty/can_dirty": {
        "class_name": "c_3_01",
        "label": "캔 + 이물질",
    },

    "dirty/glass_dirty": {
        "class_name": "c_4_03_01",
        "label": "기타유리 + 이물질",
    },

    "dirty/pet_dirty": {
        "class_name": "c_5_02_01",
        "label": "페트 + 이물질",
    },

    "dirty/plastic_dirty": {
        "class_name": "c_6_01",
        "label": "플라스틱 + 이물질",
    },


    # --------------------------------------------------------
    # 이물질 + 다중포장재
    # --------------------------------------------------------

    "dirty_packaging/pet_dirty_packaging": {
        "class_name": "c_5_01_01",
        "label": "페트 + 이물질 + 다중포장재",
    },


    # --------------------------------------------------------
    # 다중포장재
    # --------------------------------------------------------

    "packaging/glass_reuse_packaging": {
        "class_name": "c_4_01_01",
        "label": "재사용 유리 + 다중포장재",
    },

    "packaging/glass_brown_packaging": {
        "class_name": "c_4_02_01_01",
        "label": "갈색 유리 + 다중포장재",
    },

    "packaging/glass_green_packaging": {
        "class_name": "c_4_02_02_01",
        "label": "녹색 유리 + 다중포장재",
    },

    "packaging/glass_white_packaging": {
        "class_name": "c_4_02_03_01",
        "label": "백색 유리 + 다중포장재",
    },

    "packaging/pet_packaging": {
        "class_name": "c_5_01",
        "label": "페트 + 다중포장재",
    },
}


# 클래스당 저장할 샘플 수
SAMPLE_COUNT = 5

RANDOM_SEED = 42


# ============================================================
# 폰트
# ============================================================

FONT_PATH = "C:/Windows/Fonts/arial.ttf"

FONT = ImageFont.truetype(
    FONT_PATH,
    52,
)


# ============================================================
# C1 + C2 이미지 index
# ============================================================

def build_image_index():

    print("C1 + C2 이미지 검색 중...")

    image_index = {}

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG",
    ]

    for image_root in IMAGE_ROOTS:

        print(f"검색: {image_root}")

        if not image_root.exists():
            print(f"[WARNING] 폴더 없음: {image_root}")
            continue

        for ext in extensions:

            for path in image_root.rglob(ext):

                key = path.stem.lower()

                image_index[key] = path


    print(
        f"현재 확인 가능한 C1+C2 이미지: "
        f"{len(image_index):,}장"
    )

    return image_index


# ============================================================
# bbox 추출
# ============================================================

def extract_bbox(obj):

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

        return None

    return [
        x,
        y,
        x + w,
        y + h,
    ]


# ============================================================
# 이미지 열기
# ============================================================

def safe_open_image(image_path):

    try:

        image = Image.open(image_path)

        image.load()

        image = ImageOps.exif_transpose(
            image
        )

        image = image.convert("RGB")

        return image

    except Exception as e:

        print(
            f"[SKIP] 이미지 읽기 실패: "
            f"{image_path.name} / {e}"
        )

        return None


# ============================================================
# 큰 class label 표시
# ============================================================

def draw_label(
    draw,
    x,
    y,
    class_name,
    label_text,
):

    text = (
        f"{class_name} | "
        f"{label_text}"
    )

    bbox = draw.textbbox(
        (x, y),
        text,
        font=FONT,
    )

    padding = 10

    background_box = [
        bbox[0] - padding,
        bbox[1] - padding,
        bbox[2] + padding,
        bbox[3] + padding,
    ]

    draw.rectangle(
        background_box,
        fill="black",
    )

    draw.text(
        (x, y),
        text,
        fill="white",
        font=FONT,
    )


# ============================================================
# 전체 이미지 + bbox
# ============================================================

def save_full_preview(
    image,
    bbox,
    class_name,
    label_text,
    output_path,
):

    preview = image.copy()

    draw = ImageDraw.Draw(
        preview
    )

    x1, y1, x2, y2 = bbox

    draw.rectangle(
        [
            x1,
            y1,
            x2,
            y2,
        ],
        outline="red",
        width=10,
    )

    label_y = max(
        0,
        y1 - 70,
    )

    draw_label(
        draw,
        x1,
        label_y,
        class_name,
        label_text,
    )

    preview.thumbnail(
        (1800, 1800)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    preview.save(
        output_path,
        quality=95,
    )


# ============================================================
# 객체 crop
# ============================================================

def save_crop_preview(
    image,
    bbox,
    output_path,
    margin_ratio=0.35,
):

    image_width, image_height = (
        image.size
    )

    x1, y1, x2, y2 = bbox

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    margin_x = (
        bbox_width
        * margin_ratio
    )

    margin_y = (
        bbox_height
        * margin_ratio
    )

    crop_x1 = max(
        0,
        int(x1 - margin_x),
    )

    crop_y1 = max(
        0,
        int(y1 - margin_y),
    )

    crop_x2 = min(
        image_width,
        int(x2 + margin_x),
    )

    crop_y2 = min(
        image_height,
        int(y2 + margin_y),
    )

    crop = image.crop(
        (
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2,
        )
    )

    crop.thumbnail(
        (1200, 1200)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    crop.save(
        output_path,
        quality=95,
    )


# ============================================================
# 모든 대상 클래스 후보를 JSON 한 번만 읽어서 수집
# ============================================================

def collect_all_candidates(
    image_index,
):

    target_class_names = {
        config["class_name"]
        for config
        in REVIEW_CLASSES.values()
    }

    candidates = {
        class_name: []
        for class_name
        in target_class_names
    }

    json_files = list(
        LABEL_ROOT.rglob(
            "*.json"
        )
    )

    print(
        f"검색된 JSON: "
        f"{len(json_files):,}"
    )


    for index, json_path in enumerate(
        json_files,
        start=1,
    ):

        if index % 20000 == 0:

            print(
                f"JSON 처리 중: "
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

            continue


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


        # C1/C2에 없는 이미지
        if image_path is None:
            continue


        objects = data.get(
            "objects",
            [],
        )


        for obj_idx, obj in enumerate(
            objects
        ):

            class_name = obj.get(
                "class_name"
            )


            if (
                class_name
                not in target_class_names
            ):

                continue


            bbox = extract_bbox(
                obj
            )


            if bbox is None:
                continue


            candidates[
                class_name
            ].append({

                "image_name":
                    image_name,

                "image_path":
                    image_path,

                "obj_idx":
                    obj_idx,

                "bbox":
                    bbox,
            })


    return candidates


# ============================================================
# 클래스별 5개 저장
# ============================================================

def save_class_samples(
    folder_name,
    config,
    candidates,
):

    class_name = (
        config["class_name"]
    )

    label_text = (
        config["label"]
    )


    print()
    print("=" * 70)

    print(
        f"{class_name} | "
        f"{label_text}"
    )

    print("=" * 70)


    class_candidates = (
        candidates.get(
            class_name,
            []
        )
    )


    print(
        f"C1+C2 후보 bbox: "
        f"{len(class_candidates):,}"
    )


    if not class_candidates:

        print(
            "해당 클래스 이미지 없음"
        )

        return


    random.seed(
        RANDOM_SEED
    )

    random.shuffle(
        class_candidates
    )


    class_output_root = (
        OUTPUT_ROOT
        / folder_name
    )

    full_dir = (
        class_output_root
        / "full"
    )

    crop_dir = (
        class_output_root
        / "crop"
    )


    saved = 0

    used_images = set()


    for item in class_candidates:

        image_name = (
            item["image_name"]
        )


        # 같은 사진 중복 방지
        if image_name in used_images:
            continue


        image = safe_open_image(
            item["image_path"]
        )


        if image is None:
            continue


        stem = (
            Path(image_name)
            .stem
        )


        base_name = (
            f"{saved + 1:02d}_"
            f"{stem}_"
            f"obj{item['obj_idx']}"
        )


        full_path = (
            full_dir
            / f"{base_name}.jpg"
        )

        crop_path = (
            crop_dir
            / f"{base_name}.jpg"
        )


        try:

            save_full_preview(
                image=image,
                bbox=item["bbox"],
                class_name=class_name,
                label_text=label_text,
                output_path=full_path,
            )

            save_crop_preview(
                image=image,
                bbox=item["bbox"],
                output_path=crop_path,
            )

        except Exception as e:

            print(
                f"[SKIP] "
                f"{image_name} / {e}"
            )

            continue


        used_images.add(
            image_name
        )

        saved += 1


        if saved >= SAMPLE_COUNT:
            break


    print(
        f"저장 완료: "
        f"{saved}장"
    )

    print(
        f"폴더: "
        f"{class_output_root}"
    )


# ============================================================
# main
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    print(
        "LABEL_ROOT:",
        LABEL_ROOT,
    )

    print(
        "LABEL exists:",
        LABEL_ROOT.exists(),
    )


    for root in IMAGE_ROOTS:

        print(
            "IMAGE:",
            root,
            "/",
            root.exists(),
        )


    print()


    image_index = (
        build_image_index()
    )


    candidates = (
        collect_all_candidates(
            image_index
        )
    )


    print()
    print("=" * 70)
    print("후보 통계")
    print("=" * 70)


    for folder_name, config in (
        REVIEW_CLASSES.items()
    ):

        class_name = (
            config["class_name"]
        )

        print(
            f"{class_name:15} "
            f"{len(candidates.get(class_name, [])):,}"
        )


    for folder_name, config in (
        REVIEW_CLASSES.items()
    ):

        save_class_samples(
            folder_name=folder_name,
            config=config,
            candidates=candidates,
        )


    print()
    print("=" * 70)
    print("완료")
    print("=" * 70)

    print(
        f"결과:\n"
        f"{OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()
