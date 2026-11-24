from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "train_data"
TARGET_PATH = DATA_DIR / "train_target.csv"

ARTIFACTS = ROOT / "artifacts"
FEATURES_DIR = ARTIFACTS / "features"
MODELS_DIR = ARTIFACTS / "models"
FIGURES_DIR = ARTIFACTS / "figures"
RESULTS_DIR = ARTIFACTS / "results"
SPEC_PATH = ARTIFACTS / "feature_spec.json"

for _d in (ARTIFACTS, FEATURES_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

N_PARTS = 12
# Application ids grow with application date, so the last two partitions
# (ids >= 2.5M) form the out-of-time sample
OOT_ID_START = 2_500_000
SEED = 42


def raw_part_path(i: int) -> Path:
    return RAW_DIR / f"train_data_{i}.pq"
