import json
import math
from datetime import datetime, timezone

import requests

from src.config import CANDIDATES, CATCHMENT_RADIUS_KM, RAW_DIR


# ============================================================
# PUBLIC DATA SOURCES
# ============================================================

ARC_GIS_INCOME_URL = (
    "https://services7.arcgis.com/8phUg7DrlXpKgLyA/"
    "ArcGIS/rest/services/Mumbai_WFL1/FeatureServer/4/query"
)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


# ============================================================
# COMMON HELPERS
# ============================================================

def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _haversine_km(lat1, lon1, lat2, lon2):
    """
    Calculate straight-line distance between two coordinates.
    """
    radius = 6371.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(d_lambda / 2) ** 2
    )

    return 2 * radius * math.asin(math.sqrt(a))


def _candidate_rows():
    """
    Return candidates in a consistent format.
    """
    rows = []

    for candidate in CANDIDATES:
        rows.append(
            {
                "area_id": candidate["id"],
                "area": candidate["name"],
                "lat": float(candidate["lat"]),
                "lon": float(candidate["lon"]),
            }
        )

    return rows


# ============================================================
# AREA MASTER
# ============================================================

def collect_area_master():
    """
    Basic master table for the three candidate locations.
    """

    rows = []

    for candidate in CANDIDATES:
        rows.append(
            {
                "area_id": candidate["id"],
                "area": candidate["name"],
                "latitude": float(candidate["lat"]),
                "longitude": float(candidate["lon"]),
                "catchment_radius_km": CATCHMENT_RADIUS_KM,
                "source": "prototype_configuration",
                "status": "ok",
                "retrieved_at": _now_iso(),
            }
        )

    return rows


# ============================================================
# MARKET PROFILE
# ============================================================

def collect_market_profile():
    """
    Pull public Mumbai demographic / purchasing-power signals
    from the public ArcGIS layer.
    """

    rows = []

    for candidate in CANDIDATES:

        params = {
            "where": "1=1",
            "outFields": (
                "GRID_ID,"
                "ProjectedPopulationEsriIndia_TOT_P_GHSL_2025,"
                "PurchasingPowerEsriIndia_PPPC_CY,"
                "HouseholdsEsriIndia_TOTHH_CY"
            ),
            "geometry": (
                f"{candidate['lon']},{candidate['lat']}"
            ),
            "geometryType": "esriGeometryPoint",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "returnGeometry": "false",
            "f": "json",
        }

        record = {
            "area_id": candidate["id"],
            "area": candidate["name"],
            "population": None,
            "purchasing_power_proxy": None,
            "households": None,
            "grid_id": None,
            "source": ARC_GIS_INCOME_URL,
            "status": "error",
            "retrieved_at": _now_iso(),
        }

        try:
            response = requests.get(
                ARC_GIS_INCOME_URL,
                params=params,
                timeout=30,
            )
            response.raise_for_status()

            payload = response.json()
            features = payload.get("features", [])

            if features:
                attributes = features[0].get("attributes", {})

                record["population"] = attributes.get(
                    "ProjectedPopulationEsriIndia_TOT_P_GHSL_2025"
                )

                record["purchasing_power_proxy"] = attributes.get(
                    "PurchasingPowerEsriIndia_PPPC_CY"
                )

                record["households"] = attributes.get(
                    "HouseholdsEsriIndia_TOTHH_CY"
                )

                record["grid_id"] = attributes.get("GRID_ID")

                record["status"] = "ok"

            else:
                record["status"] = "no_match"

        except Exception as exc:
            record["error"] = str(exc)

        rows.append(record)

    return rows


# ============================================================
# COMPETITION
# ============================================================

def _build_competitor_query(lat, lon, radius_m):
    """
    OpenStreetMap / Overpass query for relevant retail competitors.
    """

    return f"""
    [out:json][timeout:60];

    (
      nwr["shop"="electronics"](around:{radius_m},{lat},{lon});
      nwr["shop"="mobile_phone"](around:{radius_m},{lat},{lon});
      nwr["shop"="computer"](around:{radius_m},{lat},{lon});
      nwr["shop"="department_store"](around:{radius_m},{lat},{lon});
    );

    out center tags;
    """


def collect_competition():
    """
    Collect relevant retail competitors around each candidate
    using OpenStreetMap / Overpass.

    All three candidate locations use the same query logic.
    """

    rows = []

    radius_m = int(CATCHMENT_RADIUS_KM * 1000)

    for candidate in CANDIDATES:

        query = _build_competitor_query(
            candidate["lat"],
            candidate["lon"],
            radius_m,
        )

        try:
            response = requests.post(
                OVERPASS_URL,
                data=query,
                timeout=90,
            )

            response.raise_for_status()

            payload = response.json()
            elements = payload.get("elements", [])

            candidate_rows = []

            for element in elements:

                tags = element.get("tags", {})

                name = tags.get("name")

                if not name:
                    continue

                # Do not treat Croma itself as a competitor.
                if "croma" in name.lower():
                    continue

                if element.get("type") == "node":
                    lat = element.get("lat")
                    lon = element.get("lon")

                else:
                    center = element.get("center", {})

                    lat = center.get("lat")
                    lon = center.get("lon")

                if lat is None or lon is None:
                    continue

                distance = _haversine_km(
                    float(candidate["lat"]),
                    float(candidate["lon"]),
                    float(lat),
                    float(lon),
                )

                if distance > CATCHMENT_RADIUS_KM:
                    continue

                candidate_rows.append(
                    {
                        "area_id": candidate["id"],
                        "area": candidate["name"],
                        "competitor_name": name,
                        "category": tags.get("shop"),
                        "latitude": float(lat),
                        "longitude": float(lon),
                        "distance_km": round(distance, 3),
                        "source": OVERPASS_URL,
                        "status": "ok",
                        "retrieved_at": _now_iso(),
                    }
                )

            if candidate_rows:
                rows.extend(candidate_rows)

            else:
                rows.append(
                    {
                        "area_id": candidate["id"],
                        "area": candidate["name"],
                        "competitor_name": None,
                        "category": None,
                        "latitude": None,
                        "longitude": None,
                        "distance_km": None,
                        "source": OVERPASS_URL,
                        "status": "no_match",
                        "retrieved_at": _now_iso(),
                    }
                )

        except Exception as exc:

            rows.append(
                {
                    "area_id": candidate["id"],
                    "area": candidate["name"],
                    "competitor_name": None,
                    "category": None,
                    "latitude": None,
                    "longitude": None,
                    "distance_km": None,
                    "source": OVERPASS_URL,
                    "status": "error",
                    "error": str(exc),
                    "retrieved_at": _now_iso(),
                }
            )

    return rows


# ============================================================
# ACCESSIBILITY
# ============================================================

def _build_accessibility_query(lat, lon, radius_m):
    """
    OpenStreetMap / Overpass query for public transport
    around the candidate location.

    We deliberately use the same query for every candidate.
    """

    return f"""
    [out:json][timeout:60];

    (
      nwr["railway"="station"](around:{radius_m},{lat},{lon});
      nwr["railway"="halt"](around:{radius_m},{lat},{lon});
      nwr["public_transport"="station"](around:{radius_m},{lat},{lon});
      nwr["public_transport"="stop_position"](around:{radius_m},{lat},{lon});
      nwr["amenity"="bus_station"](around:{radius_m},{lat},{lon});
      nwr["highway"="bus_stop"](around:{radius_m},{lat},{lon});
    );

    out center tags;
    """


def collect_accessibility():
    """
    Collect public-transit accessibility signals around each
    candidate location.

    Raw outputs:
      - number of transit nodes
      - nearest transit node
      - average transit distance
      - transit nodes within 500m
      - transit nodes within 1km
      - transit nodes within 2km

    These are raw signals only. The final accessibility score
    will be calculated later in the feature/engine layer.
    """

    rows = []

    radius_m = int(CATCHMENT_RADIUS_KM * 1000)

    for candidate in CANDIDATES:

        query = _build_accessibility_query(
            candidate["lat"],
            candidate["lon"],
            radius_m,
        )

        base_record = {
            "area_id": candidate["id"],
            "area": candidate["name"],
            "transit_nodes": 0,
            "transit_nodes_500m": 0,
            "transit_nodes_1km": 0,
            "transit_nodes_2km": 0,
            "nearest_transit_distance_km": None,
            "average_transit_distance_km": None,
            "source": OVERPASS_URL,
            "status": "error",
            "retrieved_at": _now_iso(),
        }

        try:

            response = requests.post(
                OVERPASS_URL,
                data=query,
                timeout=90,
            )

            response.raise_for_status()

            payload = response.json()
            elements = payload.get("elements", [])

            # ------------------------------------------------
            # Deduplicate OSM objects.
            # ------------------------------------------------

            unique_nodes = {}

            for element in elements:

                element_id = (
                    element.get("type"),
                    element.get("id"),
                )

                if element_id in unique_nodes:
                    continue

                tags = element.get("tags", {})

                # Prefer actual named transport infrastructure,
                # but allow unnamed bus stops / stop positions.
                name = tags.get("name")

                if element.get("type") == "node":

                    lat = element.get("lat")
                    lon = element.get("lon")

                else:

                    center = element.get("center", {})

                    lat = center.get("lat")
                    lon = center.get("lon")

                if lat is None or lon is None:
                    continue

                distance = _haversine_km(
                    float(candidate["lat"]),
                    float(candidate["lon"]),
                    float(lat),
                    float(lon),
                )

                if distance > CATCHMENT_RADIUS_KM:
                    continue

                unique_nodes[element_id] = {
                    "name": name,
                    "distance_km": distance,
                    "railway": tags.get("railway"),
                    "public_transport": tags.get(
                        "public_transport"
                    ),
                    "highway": tags.get("highway"),
                    "amenity": tags.get("amenity"),
                }

            distances = [
                item["distance_km"]
                for item in unique_nodes.values()
            ]

            if not distances:

                base_record["status"] = "no_match"

                rows.append(base_record)
                continue

            distances.sort()

            base_record["transit_nodes"] = len(distances)

            base_record["transit_nodes_500m"] = sum(
                d <= 0.5 for d in distances
            )

            base_record["transit_nodes_1km"] = sum(
                d <= 1.0 for d in distances
            )

            base_record["transit_nodes_2km"] = sum(
                d <= 2.0 for d in distances
            )

            base_record["nearest_transit_distance_km"] = round(
                distances[0],
                3,
            )

            base_record["average_transit_distance_km"] = round(
                sum(distances) / len(distances),
                3,
            )

            base_record["status"] = "ok"

        except Exception as exc:

            base_record["error"] = str(exc)

        rows.append(base_record)

    return rows


# ============================================================
# RENT
# ============================================================

def collect_rent():
    """
    Rent is intentionally left as a separate acquisition bucket.

    For V2 prototype, this can later be populated from a consistent
    public rental source or a manually validated assumption table.

    We do not fabricate rent values here.
    """

    rows = []

    for candidate in CANDIDATES:

        rows.append(
            {
                "area_id": candidate["id"],
                "area": candidate["name"],
                "annual_rent": None,
                "monthly_rent": None,
                "source": "pending_public_rent_source",
                "status": "pending",
                "retrieved_at": _now_iso(),
            }
        )

    return rows


# ============================================================
# NEWS SIGNALS
# ============================================================

def collect_news_signals():
    """
    News acquisition remains a placeholder at this stage.

    Later this can collect structured location-relevant signals
    from the same source logic across all candidate areas.
    """

    rows = []

    for candidate in CANDIDATES:

        rows.append(
            {
                "area_id": candidate["id"],
                "area": candidate["name"],
                "news_signal": None,
                "source": "pending_news_source",
                "status": "pending",
                "retrieved_at": _now_iso(),
            }
        )

    return rows


# ============================================================
# MASTER ACQUISITION FUNCTION
# ============================================================

def collect_public_data():
    """
    Run all acquisition modules.
    """

    return {
        "areas": collect_area_master(),
        "market_profile": collect_market_profile(),
        "competition": collect_competition(),
        "accessibility": collect_accessibility(),
        "rent": collect_rent(),
        "news_signals": collect_news_signals(),
    }


# ============================================================
# RAW PAYLOAD WRITER
# ============================================================

def write_raw_payload(data):
    """
    Save the complete acquisition output as JSON.
    """

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = RAW_DIR / "public_data_raw.json"

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path
