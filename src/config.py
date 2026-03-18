"""
Central path configuration for E320simulator.

Resolution order:
  1. DATA_PATH environment variable
  2. .env file in the project root (key=value format)
  3. Default: ~/hep/data_Run502
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv(env_file: Path) -> dict[str, str]:
    """Parse a simple key=value .env file (no external dependencies)."""
    result: dict[str, str] = {}
    try:
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        pass
    return result


# Project root is one level above this file (src/config.py → project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DOTENV_PATH = _PROJECT_ROOT / ".env"

# Resolve DATA_ROOT: env var > .env file > default
_env_vars = _load_dotenv(_DOTENV_PATH)

DATA_ROOT: Path = Path(
    os.environ.get("DATA_PATH")
    or _env_vars.get("DATA_PATH")
    or Path.home() / "hep/data_Run502"
)

# ── Derived subdirectory paths ─────────────────────────────────────────────
SIM_DIR: Path = DATA_ROOT / "simulation"
RUNS_DIR: Path = DATA_ROOT / "runs"
OUTPUTS_DIR: Path = DATA_ROOT / "outputs"
HIT_LEVEL_PARQUET: Path = DATA_ROOT / "hit_level.parquet"
HIT_LEVEL_PROCESSED: Path = DATA_ROOT / "processed" / "hit_level.parquet"
