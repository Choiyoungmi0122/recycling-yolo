from pathlib import Path
import shutil


LABEL_ROOT = Path(
    "E:/recycling_data/raw/labels/train"
)

EXCLUDED_ROOT = Path(
    "E:/recycling_data/excluded/missing_source_images/train_json"
)

EXCLUDED_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


MISSING_STEMS = [
    "c_20220715_000141",
    "c_20220817_003096",
    "c_20220908_004734",
    "c_20220922_003347",
    "c_20221020_003037",
]


# 전체 JSON index
json_index = {
    p.stem.lower(): p
    for p in LABEL_ROOT.rglob("*.json")
}


print("=" * 70)
print("MOVE MISSING-SOURCE JSON")
print("=" * 70)


moved = 0


for stem in MISSING_STEMS:

    source = json_index.get(
        stem.lower()
    )

    if source is None:
        print(
            f"[NOT FOUND] {stem}"
        )
        continue

    destination = (
        EXCLUDED_ROOT
        / source.name
    )

    print()
    print("MOVE")
    print("FROM:", source)
    print("TO  :", destination)

    shutil.move(
        str(source),
        str(destination),
    )

    moved += 1


print()
print("=" * 70)
print(f"이동 완료: {moved}개")
print("=" * 70)