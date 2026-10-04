from __future__ import annotations

from typing import Any

from src.config import CANDIDATES


def build_location_features(
    validated: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    """
    Build one canonical feature record for every configured candidate.

    Important:
    - BKC, Dadar and Vikhroli must always remain present.
    - Missing public data is represented as None.
    - We never drop a candidate just because one source failed.
    """

    # Index whatever area-level acquisition data is available.
    area_lookup = {
        str(row.get("area", "")).strip().upper(): row
        for row in validated.get("areas", [])
        if row.get("area")
    }

    result: dict[str, dict[str, Any]] = {}

    # CANDIDATES is the source of truth for the three V2 test areas.
    for candidate in CANDIDATES:
        candidate_id = candidate["id"]
        area_name = candidate["name"]
        acquired = area_lookup.get(area_name.upper(), {})

        result[candidate_id] = {
            "area": area_name,
            "latitude": candidate["lat"],
            "longitude": candidate["lon"],
            "catchment_radius_km": acquired.get(
                "catchment_radius_km", 2.0
            ),

            # Public-data fields
            "population": acquired.get("population"),
            "purchasing_power_proxy": acquired.get(
                "purchasing_power_proxy"
            ),
            "activity_proxy": acquired.get("activity_proxy"),
            "competition_count": acquired.get("competition_count"),
            "accessibility_index": acquired.get(
                "accessibility_index"
            ),
            "rent_benchmark": acquired.get("rent_benchmark"),
            "commercial_intensity": acquired.get(
                "commercial_intensity"
            ),
            "growth_signal": acquired.get("growth_signal"),

            "data_status": (
                "Public-data candidate record created; "
                "individual fields populate as sources become available."
            ),
        }

    return result
