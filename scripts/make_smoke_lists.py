from pathlib import Path
import random

DATA_ROOT = Path("E:/recycling_data")

TRAIN_FULL = DATA_ROOT / "train_has_target.txt"
VAL_FULL = DATA_ROOT / "val_has_target.txt"

TRAIN_SMOKE = DATA_ROOT / "train_smoke.txt"
VAL_SMOKE = DATA_ROOT / "val_smoke.txt"

SEED = 42
TRAIN_N = 2000
VAL_N = 500

random.seed(SEED)

train = [
    x.strip()
    for x in TRAIN_FULL.read_text(encoding="utf-8").splitlines()
    if x.strip()
]

val = [
    x.strip()
    for x in VAL_FULL.read_text(encoding="utf-8").splitlines()
    if x.strip()
]

train_smoke = random.sample(train, TRAIN_N)
val_smoke = random.sample(val, VAL_N)

TRAIN_SMOKE.write_text(
    "\n".join(train_smoke) + "\n",
    encoding="utf-8"
)

VAL_SMOKE.write_text(
    "\n".join(val_smoke) + "\n",
    encoding="utf-8"
)

print(f"Train smoke: {len(train_smoke):,}")
print(f"Val smoke  : {len(val_smoke):,}")