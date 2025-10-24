# app/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root

# Artifacts
ART_DIR_EQUITY = ROOT / "artifacts"
ART_DIR_CRYPTO = ROOT / "artifacts_crypto"  # will be created in NB20

# Optional per-ticker thresholds (equities)
TAU_MAP_PATH = ART_DIR_EQUITY / "tau_map.json"

# Defaults
DEFAULT_TAU = 0.59

# Local data
DATA_CSV = ROOT / "data" / "df_nb02.csv"
DATA_PQ  = ROOT / "data" / "df_nb02.parquet"
