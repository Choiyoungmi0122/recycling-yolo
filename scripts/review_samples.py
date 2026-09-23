from pathlib import Path
import json
import random

from PIL import Image, ImageDraw, ImageOps

# ============================================================
# 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Training 라벨 JSON - SSD
LABEL_ROOT = Path(
    "E:/recycling_data/raw/labels/train"
)

# Training 원천 이미지 - SSD
# 아래에 C_1, C_2, ... 폴더가 위치
IMAGE_ROOT = Path(
    "E:/recycling_data/raw/images/train/application_C"
)

# 결과만 C드라이브 프로젝트에 저장
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "normal_dirty_review"
)
# ============================================================
# 비교할 클래스 쌍
# glass는 dirty가 c_4_03_01 이므로 normal도 c_4_03 으로 맞춤
# ============================================================

REVIEW_CONFIG = {
    "can": {
        "normal": "c_3",
        "dirty": "c_3_01",
    },
    "glass": {
        "normal": "c_4_03",
        "dirty": "c_4_03_01",
    },
    "pet": {
        "normal": "c_5_02",
        "dirty": "c_5_02_01",
    },
    "plastic": {
        "normal": "c_6",
        "dirty": "c_6_01",
    },
}

# 각 상태별 몇 개 볼지
SAMPLE_COUNT = 5

RANDOM_SEED = 42


# ============================================================
# 이미지 index
# ============================================================

def build_image_index():
    print("현재 존재하는 이미지 검색 중...")

    image_index = {}

    extensions = [
        "*.jpg", "*.jpeg", "*.png",
        "*.JPG", "*.JPEG", "*.PNG",
    ]

    for ext in extensions:
        for path in IMAGE_ROOT.rglob(ext):

            # 확장자와 대소문자를 무시하고 파일 stem만 사용
            key = path.stem.lower()

            image_index[key] = path

    print(f"현재 확인 가능한 이미지 수: {len(image_index):,}")

    # 실제 이미지 파일명 예시 출력
    print("\n[실제 이미지 파일 예시]")
    for path in list(image_index.values())[:5]:
        print(" ", path.name)

    return image_index

# ============================================================
# bbox 추출
# annotation.coord 에서 x,y,width,height 사용
# ============================================================

def extract_bbox(obj):
    coord = obj.get("annotation", {}).get("coord", {})

    try:
        x = float(coord["x"])
        y = float(coord["y"])
        w = float(coord["width"])
        h = float(coord["height"])
    except (KeyError, TypeError, ValueError):
        return None

    return [x, y, x + w, y + h]


# ============================================================
# 현재 파일이 정상적으로 열리는지 확인
# 압축 해제 도중 생긴 불완전 jpg는 skip
# EXIF 회전 반영
# ============================================================

def safe_open_image(image_path):
    try:
        image = Image.open(image_path)
        image.load()
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return image
    except Exception as e:
        print(f"[SKIP] 이미지 읽기 실패: {image_path.name} / {e}")
        return None


# ============================================================
# bbox 그린 전체 이미지 저장
# ============================================================

def save_full_preview(image, bbox, class_name, output_path):
    preview = image.copy()
    draw = ImageDraw.Draw(preview)

    x1, y1, x2, y2 = bbox

    draw.rectangle(
        [x1, y1, x2, y2],
        outline="red",
        width=8,
    )

    draw.text(
        (x1, max(0, y1 - 30)),
        class_name,
        fill="red",
    )

    preview.thumbnail((1600, 1600))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output_path, quality=90)


# ============================================================
# bbox 주변 crop 저장
# 오염 상태 확인용으로 margin을 조금 줌
# ============================================================

def save_crop_preview(image, bbox, output_path, margin_ratio=0.35):
    w_img, h_img = image.size
    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1

    mx = bw * margin_ratio
    my = bh * margin_ratio

    cx1 = max(0, int(x1 - mx))
    cy1 = max(0, int(y1 - my))
    cx2 = min(w_img, int(x2 + mx))
    cy2 = min(h_img, int(y2 + my))

    crop = image.crop((cx1, cy1, cx2, cy2))

    # crop이 너무 작으면 확대해서 보기 좋게
    crop.thumbnail((1200, 1200))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path, quality=95)


# ============================================================
# 특정 class의 후보 수집
# 한 이미지에 target bbox가 여러 개면 각각 후보로 사용
# ============================================================

def collect_candidates(target_class, image_index):
    candidates = []

    json_files = list(LABEL_ROOT.rglob("*.json"))

    print(f"검색된 JSON 수: {len(json_files):,}")

    example_printed = False

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception:
            continue

        image_name = data.get(
            "Image",
            json_path.stem + ".jpg",
        )

        # 혹시 JSON Image에 경로가 포함돼 있어도 파일명만 사용
        image_stem = Path(image_name).stem.lower()

        image_path = image_index.get(image_stem)

        # 첫 JSON의 대응 정보 한 번 출력
        if not example_printed:
            print("\n[JSON 이미지명 예시]")
            print(" JSON :", image_name)
            print(" STEM :", image_stem)
            print(
                " MATCH:",
                image_path.name if image_path else "없음"
            )
            example_printed = True

        # C_1에 없는 이미지라면 skip
        if image_path is None:
            continue

        objects = data.get("objects", [])

        for obj_idx, obj in enumerate(objects):

            if obj.get("class_name") != target_class:
                continue

            bbox = extract_bbox(obj)

            if bbox is None:
                continue

            candidates.append({
                "json_path": json_path,
                "image_name": image_name,
                "image_path": image_path,
                "obj_idx": obj_idx,
                "bbox": bbox,
                "class_name": target_class,
            })

    return candidates

# ============================================================
# sample 저장
# ============================================================

def save_samples(material_name, state_name, target_class, image_index):
    print()
    print("=" * 70)
    print(f"{material_name.upper()} / {state_name.upper()} / {target_class}")
    print("=" * 70)

    candidates = collect_candidates(target_class, image_index)

    print(f"현재 후보 bbox 수: {len(candidates):,}")

    if len(candidates) == 0:
        print("현재 압축 해제된 이미지에서 해당 클래스 후보를 찾지 못했습니다.")
        return

    random.seed(RANDOM_SEED)
    random.shuffle(candidates)

    full_dir = OUTPUT_ROOT / material_name / state_name / "full"
    crop_dir = OUTPUT_ROOT / material_name / state_name / "crop"

    saved = 0

    # 중복 이미지가 너무 많지 않도록 image_name 기준 약간 제어
    used_image_names = set()

    for item in candidates:
        image_name = item["image_name"]

        # 가능하면 같은 이미지 중복 저장 방지
        if image_name in used_image_names:
            continue

        image = safe_open_image(item["image_path"])
        if image is None:
            continue

        bbox = item["bbox"]
        class_name = item["class_name"]

        stem = Path(image_name).stem
        base_name = f"{saved + 1:02d}_{stem}_obj{item['obj_idx']}"

        full_path = full_dir / f"{base_name}.jpg"
        crop_path = crop_dir / f"{base_name}.jpg"

        try:
            save_full_preview(image, bbox, class_name, full_path)
            save_crop_preview(image, bbox, crop_path)
        except Exception as e:
            print(f"[SKIP] 저장 실패: {image_name} / {e}")
            continue

        used_image_names.add(image_name)
        saved += 1

        if saved >= SAMPLE_COUNT:
            break

    print(f"저장 완료: {saved}개")
    print(f"full : {full_dir}")
    print(f"crop : {crop_dir}")


# ============================================================
# main
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("LABEL_ROOT:", LABEL_ROOT)
    print("LABEL exists:", LABEL_ROOT.exists())

    print("IMAGE_ROOT:", IMAGE_ROOT)
    print("IMAGE exists:", IMAGE_ROOT.exists())

    print()

    image_index = build_image_index()

    for material_name, config in REVIEW_CONFIG.items():
        save_samples(
            material_name=material_name,
            state_name="normal",
            target_class=config["normal"],
            image_index=image_index,
        )

        save_samples(
            material_name=material_name,
            state_name="dirty",
            target_class=config["dirty"],
            image_index=image_index,
        )


if __name__ == "__main__":
    main()
