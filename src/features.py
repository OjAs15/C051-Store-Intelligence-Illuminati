import math
from collections import defaultdict

from src.acquisition import collect_public_data
from src.config import CATCHMENT_RADIUS_KM, RAW_DIR


# ============================================================
# HELPERS
# ============================================================

def _safe_float(value):
    """
    Convert a value to float safely.
    Return None when conversion is not possible.
    """
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value):
    """
    Convert a value to int safely.
    """
    if value is None:
        return None

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _round(value, digits=4):
    """
    Round numeric values while preserving None.
    """
    if value is None:
        return None

    return round(float(value), digits)


def _catchment_area_sqkm(radius_km):
    """
    Area of a circular catchment.
    """
    return math.pi * (radius_km ** 2)


# ============================================================
# COMPETITION AGGREGATION
# ============================================================

def _aggregate_competition(rows):
    """
    Convert competitor-level observations into one row per candidate.

    Raw competitor observations are transformed into:
      - competitor count
      - electronics competitor count
      - mobile phone competitor count
      - department store count
      - computer store count
      - nearest competitor distance
      - distance-decayed competitor pressure
    """

    aggregate = defaultdict(
        lambda: {
            "competitor_count": 0,
            "electronics_competitor_count": 0,
            "mobile_competitor_count": 0,
            "computer_competitor_count": 0,
            "department_store_competitor_count": 0,
            "nearest_competitor_distance_km": None,
            "competitor_pressure": 0.0,
        }
    )

    for row in rows:

        area_id = row.get("area_id")

        if not area_id:
            continue

        if row.get("status") != "ok":
            continue

        distance = _safe_float(
            row.get("distance_km")
        )

        if distance is None:
            continue

        category = (
            str(row.get("category") or "")
            .strip()
            .lower()
        )

        aggregate[area_id]["competitor_count"] += 1

        if category == "electronics":
            aggregate[area_id][
                "electronics_competitor_count"
            ] += 1

        elif category == "mobile_phone":
            aggregate[area_id][
                "mobile_competitor_count"
            ] += 1

        elif category == "computer":
            aggregate[area_id][
                "computer_competitor_count"
            ] += 1

        elif category == "department_store":
            aggregate[area_id][
                "department_store_competitor_count"
            ] += 1

        nearest = aggregate[area_id][
            "nearest_competitor_distance_km"
        ]

        if nearest is None or distance < nearest:
            aggregate[area_id][
                "nearest_competitor_distance_km"
            ] = distance

        # --------------------------------------------
        # Distance-decay pressure
        #
        # Competitors closer to the candidate exert
        # greater competitive pressure.
        # --------------------------------------------

        decay = math.exp(-distance / 1.0)

        aggregate[area_id][
            "competitor_pressure"
        ] += decay

    # Round final values.
    for area_id, values in aggregate.items():
        values["competitor_pressure"] = _round(
            values["competitor_pressure"],
            4,
        )

        values["nearest_competitor_distance_km"] = _round(
            values["nearest_competitor_distance_km"],
            3,
        )

    return aggregate


# ============================================================
# ACCESSIBILITY AGGREGATION
# ============================================================

def _aggregate_accessibility(rows):
    """
    Prepare candidate-level accessibility features.

    We retain raw transit signals and add only simple
    structural derivatives such as density.
    """

    aggregate = {}

    catchment_area = _catchment_area_sqkm(
        CATCHMENT_RADIUS_KM
    )

    for row in rows:

        area_id = row.get("area_id")

        if not area_id:
            continue

        transit_nodes = _safe_int(
            row.get("transit_nodes")
        ) or 0

        transit_nodes_500m = _safe_int(
            row.get("transit_nodes_500m")
        ) or 0

        transit_nodes_1km = _safe_int(
            row.get("transit_nodes_1km")
        ) or 0

        transit_nodes_2km = _safe_int(
            row.get("transit_nodes_2km")
        ) or 0

        nearest_distance = _safe_float(
            row.get("nearest_transit_distance_km")
        )

        average_distance = _safe_float(
            row.get("average_transit_distance_km")
        )

        aggregate[area_id] = {
            "transit_nodes": transit_nodes,
            "transit_nodes_500m": transit_nodes_500m,
            "transit_nodes_1km": transit_nodes_1km,
            "transit_nodes_2km": transit_nodes_2km,
            "nearest_transit_distance_km": _round(
                nearest_distance,
                3,
            ),
            "average_transit_distance_km": _round(
                average_distance,
                3,
            ),
            "transit_density_per_sqkm": _round(
                transit_nodes / catchment_area,
                4,
            ),
        }

    return aggregate


# ============================================================
# MARKET DATA PREPARATION
# ============================================================

def _prepare_market(rows):
    """
    Standardize public market-profile data.
    """

    result = {}

    for row in rows:

        area_id = row.get("area_id")

        if not area_id:
            continue

        population = _safe_float(
            row.get("population")
        )

        purchasing_power = _safe_float(
            row.get("purchasing_power_proxy")
        )

        households = _safe_float(
            row.get("households")
        )

        population_per_household = None

        if (
            population is not None
            and households is not None
            and households > 0
        ):
            population_per_household = (
                population / households
            )

        result[area_id] = {
            "population": population,
            "purchasing_power_proxy": purchasing_power,
            "households": households,
            "population_per_household": _round(
                population_per_household,
                3,
            ),
            "market_data_status": row.get(
                "status",
                "unknown",
            ),
            "market_data_source": row.get(
                "source"
            ),
        }

    return result


# ============================================================
# MAIN FEATURE BUILDER
# ============================================================

def build_candidate_features(data=None):
    """
    Build one clean candidate-level feature table.

    Each candidate gets exactly one row.

    This function deliberately does NOT calculate:
      - Market Potential Score
      - Demand Prediction
      - Demand Capture Score
      - Economics Score
      - Cannibalisation Score
      - Final Recommendation

    Those belong to downstream decision engines.
    """

    if data is None:
        data = collect_public_data()

    areas = data.get("areas", [])
    market_rows = data.get("market_profile", [])
    competition_rows = data.get("competition", [])
    accessibility_rows = data.get("accessibility", [])
    rent_rows = data.get("rent", [])
    news_rows = data.get("news_signals", [])

    competition = _aggregate_competition(
        competition_rows
    )

    accessibility = _aggregate_accessibility(
        accessibility_rows
    )

    market = _prepare_market(
        market_rows
    )

    # --------------------------------------------------------
    # Rent lookup
    # --------------------------------------------------------

    rent_lookup = {}

    for row in rent_rows:

        area_id = row.get("area_id")

        if not area_id:
            continue

        rent_lookup[area_id] = {
            "monthly_rent": _safe_float(
                row.get("monthly_rent")
            ),
            "annual_rent": _safe_float(
                row.get("annual_rent")
            ),
            "rent_status": row.get(
                "status",
                "unknown",
            ),
        }

    # --------------------------------------------------------
    # News lookup
    # --------------------------------------------------------

    news_lookup = {}

    for row in news_rows:

        area_id = row.get("area_id")

        if not area_id:
            continue

        news_lookup[area_id] = {
            "news_signal": row.get(
                "news_signal"
            ),
            "news_status": row.get(
                "status",
                "unknown",
            ),
        }

    # --------------------------------------------------------
    # Construct final rows
    # --------------------------------------------------------

    feature_rows = []

    for area in areas:

        area_id = area.get("area_id")

        if not area_id:
            continue

        comp = competition.get(
            area_id,
            {
                "competitor_count": 0,
                "electronics_competitor_count": 0,
                "mobile_competitor_count": 0,
                "computer_competitor_count": 0,
                "department_store_competitor_count": 0,
                "nearest_competitor_distance_km": None,
                "competitor_pressure": 0.0,
            },
        )

        access = accessibility.get(
            area_id,
            {
                "transit_nodes": 0,
                "transit_nodes_500m": 0,
                "transit_nodes_1km": 0,
                "transit_nodes_2km": 0,
                "nearest_transit_distance_km": None,
                "average_transit_distance_km": None,
                "transit_density_per_sqkm": 0.0,
            },
        )

        market_data = market.get(
            area_id,
            {
                "population": None,
                "purchasing_power_proxy": None,
                "households": None,
                "population_per_household": None,
                "market_data_status": "missing",
                "market_data_source": None,
            },
        )

        rent = rent_lookup.get(
            area_id,
            {
                "monthly_rent": None,
                "annual_rent": None,
                "rent_status": "pending",
            },
        )

        news = news_lookup.get(
            area_id,
            {
                "news_signal": None,
                "news_status": "pending",
            },
        )

        population = market_data["population"]

        households = market_data["households"]

        catchment_area = _catchment_area_sqkm(
            CATCHMENT_RADIUS_KM
        )

        population_density = None

        if population is not None:
            population_density = (
                population / catchment_area
            )

        household_density = None

        if households is not None:
            household_density = (
                households / catchment_area
            )

        feature_rows.append(
            {
                "area_id": area_id,
                "area": area.get("area"),
                "latitude": _safe_float(
                    area.get("latitude")
                ),
                "longitude": _safe_float(
                    area.get("longitude")
                ),

                # ----------------------------------------
                # Market
                # ----------------------------------------

                "population": _round(
                    population,
                    2,
                ),
                "population_density_per_sqkm": _round(
                    population_density,
                    2,
                ),
                "households": _round(
                    households,
                    2,
                ),
                "household_density_per_sqkm": _round(
                    household_density,
                    2,
                ),
                "purchasing_power_proxy": _round(
                    market_data[
                        "purchasing_power_proxy"
                    ],
                    2,
                ),
                "population_per_household": market_data[
                    "population_per_household"
                ],

                # ----------------------------------------
                # Competition
                # ----------------------------------------

                "competitor_count": comp[
                    "competitor_count"
                ],
                "electronics_competitor_count": comp[
                    "electronics_competitor_count"
                ],
                "mobile_competitor_count": comp[
                    "mobile_competitor_count"
                ],
                "computer_competitor_count": comp[
                    "computer_competitor_count"
                ],
                "department_store_competitor_count": comp[
                    "department_store_competitor_count"
                ],
                "nearest_competitor_distance_km": comp[
                    "nearest_competitor_distance_km"
                ],
                "competitor_pressure": comp[
                    "competitor_pressure"
                ],

                # ----------------------------------------
                # Accessibility
                # ----------------------------------------

                "transit_nodes": access[
                    "transit_nodes"
                ],
                "transit_nodes_500m": access[
                    "transit_nodes_500m"
                ],
                "transit_nodes_1km": access[
                    "transit_nodes_1km"
                ],
                "transit_nodes_2km": access[
                    "transit_nodes_2km"
                ],
                "nearest_transit_distance_km": access[
                    "nearest_transit_distance_km"
                ],
                "average_transit_distance_km": access[
                    "average_transit_distance_km"
                ],
                "transit_density_per_sqkm": access[
                    "transit_density_per_sqkm"
                ],

                # ----------------------------------------
                # Economics
                # ----------------------------------------

                "monthly_rent": rent[
                    "monthly_rent"
                ],
                "annual_rent": rent[
                    "annual_rent"
                ],
                "rent_status": rent[
                    "rent_status"
                ],

                # ----------------------------------------
                # External signals
                # ----------------------------------------

                "news_signal": news[
                    "news_signal"
                ],
                "news_status": news[
                    "news_status"
                ],

                # ----------------------------------------
                # Metadata
                # ----------------------------------------

                "catchment_radius_km": CATCHMENT_RADIUS_KM,
                "catchment_area_sqkm": _round(
                    catchment_area,
                    4,
                ),
            }
        )

    return feature_rows


# ============================================================
# CSV WRITER
# ============================================================

def write_feature_csv(rows, output_path=None):
    """
    Save the feature table as CSV.
    """

    import csv

    if output_path is None:

        output_path = (
            RAW_DIR.parent
            / "processed"
            / "candidate_features.csv"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise ValueError(
            "Cannot write feature CSV: no rows supplied."
        )

    fieldnames = list(rows[0].keys())

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    return output_path


# ============================================================
# END-TO-END FEATURE PIPELINE
# ============================================================

def build_and_write_features():

    data = collect_public_data()

    rows = build_candidate_features(data)

    output_path = write_feature_csv(rows)

    return {
        "rows": rows,
        "output_path": output_path,
    }
