from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
COUNTRY_CONFIG_PATH = BASE_DIR / "app" / "config" / "countries.yaml"


class CountryProfile(BaseModel):
    name: str
    output_width_px: int
    output_height_px: int
    max_file_size_mb: int
    allowed_extensions: list[str] = Field(default_factory=list)
    max_age_days: int
    min_input_width_px: int
    min_input_height_px: int
    min_eye_distance_px: int
    min_face_height_px: int
    eye_distance_fraction_of_width: float
    eye_height_fraction_of_height: float
    extra_headroom_fraction: float = 0.03
    extra_torso_fraction: float = 0.05
    hair_margin_fraction: float = 0.24
    max_roll_degrees: float
    max_yaw_ratio: float
    max_pitch_ratio: float
    min_background_brightness: float
    max_background_saturation: float
    max_background_stddev: float
    min_blur_score: float
    min_even_lighting_score: float


class CountrySettings(BaseModel):
    default_country: str
    countries: dict[str, CountryProfile]


@lru_cache(maxsize=1)
def load_country_settings() -> CountrySettings:
    with COUNTRY_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return CountrySettings(**raw)


def get_country_profile(country_code: str) -> CountryProfile:
    settings = load_country_settings()
    code = (country_code or settings.default_country).strip().upper()
    if code not in settings.countries:
        code = settings.default_country
    return settings.countries[code]


def list_country_profiles() -> dict[str, CountryProfile]:
    return load_country_settings().countries
