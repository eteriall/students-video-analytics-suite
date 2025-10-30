from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    max_unique_profiles: int = _get_int("MAX_UNIQUE_PROFILES", 30)
    ambiguous_distance_margin: float = _get_float("AMBIGUOUS_DISTANCE_MARGIN", 0.05)


settings = Settings()
