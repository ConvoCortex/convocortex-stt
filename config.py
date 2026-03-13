import sys
import tomllib
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.toml"


def load() -> dict:
    if not CONFIG_PATH.exists():
        print(f"[config] config.toml not found at {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(CONFIG_PATH, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        print(f"[config] Failed to parse config.toml: {e}", file=sys.stderr)
        sys.exit(1)
