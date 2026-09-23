from pathlib import Path
from collections import Counter
import json

import pandas as pd
from tqdm import tqdm


# ============================================================
# 1. 프로젝트 경로
# ============================================================

# 현재 파일:
# RECYCLING/scripts/analyze_labels.py
#
# parents[0] -> scripts
# parents[1] -> RECYCLING

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "TL_application_C"
VAL_DIR = PROJECT_ROOT / "data" / "raw" / "VL_application_C"

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 분석 대상 클래스 정의
# ============================================================
#
# material
#   CAN
#   GLASS
#   PET
#   PLASTIC
#
# status
#   NORMAL
#       현재 상태 그대로 정상 분리배출 가능
#
#   DIRTY
#       이물질 제거 / 세척 필요
#
#   PACKAGING
#       다중포장재 제거 필요
#
#   DIRTY_PACKAGING
#       이물질 + 다중포장재 모두 존재
#
# recycling_state
#   READY
#       현재 상태로 배출 가능
#
#   NEEDS_PREPROCESS
#       세척 또는 포장재 제거 후 배출
# ============================================================

CLASS_INFO = {

    # ========================================================
    # CAN
    # ========================================================

    "c_3": {
        "material": "CAN",
        "status": "NORMAL",
        "label": "캔류",
        "recycling_state": "READY",
    },

    "c_3_01": {
        "material": "CAN",
        "status": "DIRTY",
        "label": "캔 + 이물질",
        "recycling_state": "NEEDS_PREPROCESS",
    },


    # ========================================================
    # GLASS - NORMAL
    # ========================================================

    "c_4_01_02": {
        "material": "GLASS",
        "status": "NORMAL",
        "label": "재사용 유리 (소주병+맥주병)",
        "recycling_state": "READY",
    },

    "c_4_02_01_02": {
        "material": "GLASS",
        "status": "NORMAL",
        "label": "갈색 유리",
        "recycling_state": "READY",
    },

    "c_4_02_02_02": {
        "material": "GLASS",
        "status": "NORMAL",
        "label": "녹색 유리",
        "recycling_state": "READY",
    },

    "c_4_02_03_02": {
        "material": "GLASS",
        "status": "NORMAL",
        "label": "백색 유리",
        "recycling_state": "READY",
    },

    "c_4_03": {
        "material": "GLASS",
        "status": "NORMAL",
        "label": "기타 유리",
        "recycling_state": "READY",
    },


    # ========================================================
    # GLASS - DIRTY
    # ========================================================

    "c_4_03_01": {
        "material": "GLASS",
        "status": "DIRTY",
        "label": "기타 유리 + 이물질",
        "recycling_state": "NEEDS_PREPROCESS",
    },


    # ========================================================
    # GLASS - PACKAGING
    # ========================================================

    "c_4_01_01": {
        "material": "GLASS",
        "status": "PACKAGING",
        "label": "재사용 유리 (소주병+맥주병) + 다중포장재",
        "recycling_state": "NEEDS_PREPROCESS",
    },

    "c_4_02_01_01": {
        "material": "GLASS",
        "status": "PACKAGING",
        "label": "갈색 유리 + 다중포장재",
        "recycling_state": "NEEDS_PREPROCESS",
    },

    "c_4_02_02_01": {
        "material": "GLASS",
        "status": "PACKAGING",
        "label": "녹색 유리 + 다중포장재",
        "recycling_state": "NEEDS_PREPROCESS",
    },

    "c_4_02_03_01": {
        "material": "GLASS",
        "status": "PACKAGING",
        "label": "백색 유리 + 다중포장재",
        "recycling_state": "NEEDS_PREPROCESS",
    },


    # ========================================================
    # PET
    # ========================================================

    "c_5_02": {
        "material": "PET",
        "status": "NORMAL",
        "label": "페트",
        "recycling_state": "READY",
    },

    "c_5_02_01": {
        "material": "PET",
        "status": "DIRTY",
        "label": "페트 + 이물질",
        "recycling_state": "NEEDS_PREPROCESS",
    },

    "c_5_01": {
        "material": "PET",
        "status": "PACKAGING",
        "label": "페트 + 다중포장재",
        "recycling_state": "NEEDS_PREPROCESS",
    },

    "c_5_01_01": {
        "material": "PET",
        "status": "DIRTY_PACKAGING",
        "label": "페트 + 이물질 + 다중포장재",
        "recycling_state": "NEEDS_PREPROCESS",
    },


    # ========================================================
    # PLASTIC
    # ========================================================

    "c_6": {
        "material": "PLASTIC",
        "status": "NORMAL",
        "label": "플라스틱",
        "recycling_state": "READY",
    },

    "c_6_01": {
        "material": "PLASTIC",
        "status": "DIRTY",
        "label": "플라스틱 + 이물질",
        "recycling_state": "NEEDS_PREPROCESS",
    },
}


MATERIAL_ORDER = [
    "CAN",
    "GLASS",
    "PET",
    "PLASTIC",
]

STATUS_ORDER = [
    "NORMAL",
    "DIRTY",
    "PACKAGING",
    "DIRTY_PACKAGING",
]


# ============================================================
# 3. 경로 확인
# ============================================================

def check_directories():

    print("=" * 70)
    print("Dataset path check")
    print("=" * 70)

    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"TRAIN_DIR    : {TRAIN_DIR}")
    print(f"VAL_DIR      : {VAL_DIR}")
    print(f"REPORT_DIR   : {REPORT_DIR}")

    print()
    print(f"TRAIN exists : {TRAIN_DIR.exists()}")
    print(f"VAL exists   : {VAL_DIR.exists()}")

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training directory not found:\n{TRAIN_DIR}"
        )

    if not VAL_DIR.exists():
        raise FileNotFoundError(
            f"Validation directory not found:\n{VAL_DIR}"
        )


# ============================================================
# 4. JSON 하나 분석
# ============================================================

def load_json(json_path):

    with open(
        json_path,
        "r",
        encoding="utf-8-sig",
    ) as f:

        return json.load(f)


# ============================================================
# 5. Split 분석
# ============================================================

def analyze_split(folder, split_name):

    json_files = list(folder.rglob("*.json"))

    print()
    print("=" * 70)
    print(f"{split_name.upper()} ANALYSIS")
    print("=" * 70)

    print(f"JSON files: {len(json_files):,}")

    # --------------------------------------------------------
    # 전체 데이터셋 클래스 통계
    # --------------------------------------------------------

    all_class_bbox_count = Counter()
    all_class_image_count = Counter()

    # --------------------------------------------------------
    # 우리가 사용할 클래스 통계
    # --------------------------------------------------------

    target_class_bbox_count = Counter()
    target_class_image_count = Counter()

    material_bbox_count = Counter()
    material_image_count = Counter()

    status_bbox_count = Counter()
    status_image_count = Counter()

    material_status_bbox_count = Counter()
    material_status_image_count = Counter()

    recycling_state_bbox_count = Counter()
    recycling_state_image_count = Counter()

    # --------------------------------------------------------
    # 이미지 구성
    # --------------------------------------------------------

    target_any_count = 0
    target_only_count = 0
    target_with_other_count = 0
    no_target_count = 0

    material_num_per_image = Counter()

    # --------------------------------------------------------
    # 메타데이터
    # --------------------------------------------------------

    place_count = Counter()
    daynight_count = Counter()
    method_count = Counter()

    # --------------------------------------------------------
    # 이미지별 목록
    # --------------------------------------------------------

    image_rows = []

    # --------------------------------------------------------
    # JSON 오류
    # --------------------------------------------------------

    error_rows = []

    # ========================================================
    # 모든 JSON 분석
    # ========================================================

    for json_path in tqdm(
        json_files,
        desc=f"Analyzing {split_name}",
    ):

        try:

            data = load_json(json_path)

        except Exception as e:

            error_rows.append({
                "split": split_name,
                "json_file": str(json_path),
                "error": str(e),
            })

            continue


        # ----------------------------------------------------
        # 객체 목록
        # ----------------------------------------------------

        objects = data.get("objects", [])

        labels = []

        for obj in objects:

            class_name = obj.get("class_name")

            if class_name:
                labels.append(class_name)


        unique_labels = set(labels)


        # ----------------------------------------------------
        # 전체 클래스 통계
        # ----------------------------------------------------

        all_class_bbox_count.update(labels)

        for label in unique_labels:
            all_class_image_count[label] += 1


        # ----------------------------------------------------
        # 현재 이미지에서 대상 클래스만 추출
        # ----------------------------------------------------

        target_labels = [
            label
            for label in labels
            if label in CLASS_INFO
        ]

        unique_target_labels = set(target_labels)


        image_materials = set()
        image_statuses = set()
        image_material_statuses = set()
        image_recycling_states = set()


        # ----------------------------------------------------
        # 대상 객체 통계
        # ----------------------------------------------------

        for class_name in target_labels:

            info = CLASS_INFO[class_name]

            material = info["material"]
            status = info["status"]
            recycling_state = info["recycling_state"]


            # bbox 수
            target_class_bbox_count[class_name] += 1

            material_bbox_count[material] += 1

            status_bbox_count[status] += 1

            material_status_bbox_count[
                (material, status)
            ] += 1

            recycling_state_bbox_count[
                recycling_state
            ] += 1


            # 이미지 단위 집계를 위한 set
            image_materials.add(material)

            image_statuses.add(status)

            image_material_statuses.add(
                (material, status)
            )

            image_recycling_states.add(
                recycling_state
            )


        # ----------------------------------------------------
        # 이미지 단위 클래스 카운트
        # ----------------------------------------------------

        for class_name in unique_target_labels:
            target_class_image_count[class_name] += 1


        for material in image_materials:
            material_image_count[material] += 1


        for status in image_statuses:
            status_image_count[status] += 1


        for pair in image_material_statuses:
            material_status_image_count[pair] += 1


        for state in image_recycling_states:
            recycling_state_image_count[state] += 1


        # ----------------------------------------------------
        # 이미지에 target이 있는지
        # ----------------------------------------------------

        if target_labels:

            target_any_count += 1

            # 이미지 안에 존재하는 모든 bbox가
            # 우리가 사용하는 4종 클래스인지 확인

            if all(
                label in CLASS_INFO
                for label in labels
            ):

                target_only_count += 1

                target_only = True

            else:

                target_with_other_count += 1

                target_only = False

        else:

            no_target_count += 1

            target_only = False


        # ----------------------------------------------------
        # 이미지 속 대상 재질 종류 수
        # ----------------------------------------------------

        material_num_per_image[
            len(image_materials)
        ] += 1


        # ----------------------------------------------------
        # metadata
        # ----------------------------------------------------

        info = data.get("Info", {})

        place = info.get(
            "PLACE",
            "MISSING",
        )

        daynight = info.get(
            "DAY/NIGHT",
            "MISSING",
        )

        method = info.get(
            "METHOD",
            "MISSING",
        )

        device = info.get(
            "DEVICE",
            "MISSING",
        )

        resolution = info.get(
            "RESOLUTION",
            "MISSING",
        )

        iso = info.get(
            "ISO",
            "MISSING",
        )

        exposure = info.get(
            "EXPOSURE",
            "MISSING",
        )


        place_count[place] += 1
        daynight_count[daynight] += 1
        method_count[method] += 1


        # ----------------------------------------------------
        # 원본 이미지 파일명
        # ----------------------------------------------------

        image_name = data.get(
            "Image",
            json_path.stem,
        )


        # ----------------------------------------------------
        # 이미지별 inventory
        # ----------------------------------------------------

        image_rows.append({

            "split": split_name,

            "json_file": json_path.name,

            "image_file": image_name,

            "has_target":
                len(target_labels) > 0,

            "target_only":
                target_only,

            "materials":
                "|".join(sorted(image_materials)),

            "statuses":
                "|".join(sorted(image_statuses)),

            "target_classes":
                "|".join(sorted(unique_target_labels)),

            "all_classes":
                "|".join(sorted(unique_labels)),

            "target_bbox_count":
                len(target_labels),

            "total_bbox_count":
                len(labels),

            "material_count":
                len(image_materials),

            "PLACE":
                place,

            "DAY_NIGHT":
                daynight,

            "METHOD":
                method,

            "DEVICE":
                device,

            "RESOLUTION":
                resolution,

            "ISO":
                iso,

            "EXPOSURE":
                exposure,
        })


    # ========================================================
    # 6. 전체 클래스 분포
    # ========================================================

    all_class_rows = []

    all_classes = sorted(
        all_class_bbox_count.keys()
    )

    for class_name in all_classes:

        all_class_rows.append({
            "split":
                split_name,

            "class_name":
                class_name,

            "image_count":
                all_class_image_count[class_name],

            "bbox_count":
                all_class_bbox_count[class_name],
        })


    all_class_df = pd.DataFrame(
        all_class_rows
    )


    # ========================================================
    # 7. 대상 세부 클래스 분포
    # ========================================================

    class_rows = []

    for class_name, info in CLASS_INFO.items():

        class_rows.append({

            "split":
                split_name,

            "material":
                info["material"],

            "status":
                info["status"],

            "recycling_state":
                info["recycling_state"],

            "class_name":
                class_name,

            "label":
                info["label"],

            "image_count":
                target_class_image_count[class_name],

            "bbox_count":
                target_class_bbox_count[class_name],
        })


    class_df = pd.DataFrame(
        class_rows
    )


    # ========================================================
    # 8. Material × Status
    # ========================================================

    material_status_rows = []

    for material in MATERIAL_ORDER:

        for status in STATUS_ORDER:

            key = (
                material,
                status,
            )

            image_count = (
                material_status_image_count[key]
            )

            bbox_count = (
                material_status_bbox_count[key]
            )

            if (
                image_count == 0
                and bbox_count == 0
            ):
                continue


            material_status_rows.append({

                "split":
                    split_name,

                "material":
                    material,

                "status":
                    status,

                "image_count":
                    image_count,

                "bbox_count":
                    bbox_count,
            })


    material_status_df = pd.DataFrame(
        material_status_rows
    )


    # ========================================================
    # 9. Material
    # ========================================================

    material_rows = []

    for material in MATERIAL_ORDER:

        material_rows.append({

            "split":
                split_name,

            "material":
                material,

            "image_count":
                material_image_count[material],

            "bbox_count":
                material_bbox_count[material],
        })


    material_df = pd.DataFrame(
        material_rows
    )


    # ========================================================
    # 10. Status
    # ========================================================

    status_rows = []

    for status in STATUS_ORDER:

        image_count = (
            status_image_count[status]
        )

        bbox_count = (
            status_bbox_count[status]
        )

        if (
            image_count == 0
            and bbox_count == 0
        ):
            continue


        status_rows.append({

            "split":
                split_name,

            "status":
                status,

            "image_count":
                image_count,

            "bbox_count":
                bbox_count,
        })


    status_df = pd.DataFrame(
        status_rows
    )


    # ========================================================
    # 11. 정상 / 전처리 필요
    # ========================================================

    recycling_rows = []

    for state in [
        "READY",
        "NEEDS_PREPROCESS",
    ]:

        recycling_rows.append({

            "split":
                split_name,

            "recycling_state":
                state,

            "image_count":
                recycling_state_image_count[state],

            "bbox_count":
                recycling_state_bbox_count[state],
        })


    recycling_df = pd.DataFrame(
        recycling_rows
    )


    # ========================================================
    # 12. 이미지 inventory
    # ========================================================

    image_df = pd.DataFrame(
        image_rows
    )


    # ========================================================
    # 13. Error
    # ========================================================

    error_df = pd.DataFrame(
        error_rows
    )


    # ========================================================
    # 14. 콘솔 출력
    # ========================================================

    print()
    print("-" * 70)
    print("TARGET CLASS SUMMARY")
    print("-" * 70)

    print(
        class_df[
            [
                "material",
                "status",
                "class_name",
                "label",
                "image_count",
                "bbox_count",
            ]
        ].to_string(index=False)
    )


    print()
    print("-" * 70)
    print("MATERIAL × STATUS")
    print("-" * 70)

    print(
        material_status_df.to_string(
            index=False
        )
    )


    print()
    print("-" * 70)
    print("MATERIAL SUMMARY")
    print("-" * 70)

    print(
        material_df.to_string(
            index=False
        )
    )


    print()
    print("-" * 70)
    print("STATUS SUMMARY")
    print("-" * 70)

    print(
        status_df.to_string(
            index=False
        )
    )


    print()
    print("-" * 70)
    print("RECYCLING STATE")
    print("-" * 70)

    print(
        recycling_df.to_string(
            index=False
        )
    )


    print()
    print("-" * 70)
    print("IMAGE COMPOSITION")
    print("-" * 70)

    print(
        f"4종 중 하나 이상 포함 : "
        f"{target_any_count:,}"
    )

    print(
        f"4종만 포함            : "
        f"{target_only_count:,}"
    )

    print(
        f"4종 + 다른 품목 포함  : "
        f"{target_with_other_count:,}"
    )

    print(
        f"4종 없음              : "
        f"{no_target_count:,}"
    )


    print()
    print(
        "대상 재질 개수별 이미지"
    )

    for n in sorted(
        material_num_per_image.keys()
    ):

        print(
            f"{n}종 : "
            f"{material_num_per_image[n]:,}장"
        )


    print()
    print("PLACE:")
    print(dict(place_count))

    print()
    print("DAY/NIGHT:")
    print(dict(daynight_count))

    print()
    print("METHOD:")
    print(dict(method_count))

    print()
    print(
        f"JSON read errors: "
        f"{len(error_rows):,}"
    )


    return {

        "all_class":
            all_class_df,

        "class":
            class_df,

        "material_status":
            material_status_df,

        "material":
            material_df,

        "status":
            status_df,

        "recycling":
            recycling_df,

        "images":
            image_df,

        "errors":
            error_df,
    }


# ============================================================
# 15. main
# ============================================================

def main():

    check_directories()

    train_results = analyze_split(
        TRAIN_DIR,
        "train",
    )

    val_results = analyze_split(
        VAL_DIR,
        "val",
    )


    # ========================================================
    # Train + Validation 합치기
    # ========================================================

    report_keys = [
        "all_class",
        "class",
        "material_status",
        "material",
        "status",
        "recycling",
        "images",
    ]


    filenames = {

        "all_class":
            "all_class_summary.csv",

        "class":
            "target_class_summary.csv",

        "material_status":
            "material_status_summary.csv",

        "material":
            "material_summary.csv",

        "status":
            "status_summary.csv",

        "recycling":
            "recycling_state_summary.csv",

        "images":
            "image_inventory.csv",
    }


    for key in report_keys:

        combined_df = pd.concat(
            [
                train_results[key],
                val_results[key],
            ],
            ignore_index=True,
        )

        output_path = (
            REPORT_DIR
            / filenames[key]
        )

        combined_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )


    # ========================================================
    # JSON 오류 저장
    # ========================================================

    errors = pd.concat(
        [
            train_results["errors"],
            val_results["errors"],
        ],
        ignore_index=True,
    )


    if not errors.empty:

        errors.to_csv(
            REPORT_DIR
            / "json_errors.csv",
            index=False,
            encoding="utf-8-sig",
        )


    # ========================================================
    # 완료
    # ========================================================

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Reports saved to:\n"
        f"{REPORT_DIR}"
    )

    print()

    for filename in filenames.values():
        print(f"- {filename}")


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    main()