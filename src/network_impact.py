import csv
import math
from pathlib import Path

import src.config as config


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "processed"
    / "candidate_features.csv"
)

CAPTURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "processed"
    / "demand_capture.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "processed"
    / "network_impact.csv"
)


# ============================================================
# NETWORK MODEL CONFIGURATION
# ============================================================

# Distance-decay parameter.
#
# This is a prototype modelling assumption, not an observed
# Croma parameter.
#
# Smaller lambda -> faster decay with distance.
# Larger lambda  -> slower decay with distance.

DISTANCE_DECAY_KM = 1.0


# Used only when the existing-store configuration contains
# no historical performance measure.
#
# Equal weighting is preferable to inventing store-sales data.

DEFAULT_EXISTING_STORE_WEIGHT = 1.0


# Final network-impact proxy:
#
#   70% = capturable demand
#   30% = protection from cannibalisation
#
# This is a decision-model assumption for the POC.

CAPTURE_WEIGHT = 0.70
CANNIBALISATION_PROTECTION_WEIGHT = 0.30


# ============================================================
# HELPERS
# ============================================================

def _to_float(value):
    """
    Safely convert a value to float.
    """

    if value is None or value == "":
        return None

    try:
        return float(value)

    except (TypeError, ValueError):
        return None


def _min_max_normalize(
    values,
    low=10.0,
    high=100.0,
):
    """
    Min-max normalize values.

    Identical values receive the midpoint because the feature
    does not differentiate candidates.
    """

    valid = [
        value
        for value in values
        if value is not None
    ]

    if not valid:
        return [
            None
            for _ in values
        ]

    minimum = min(valid)
    maximum = max(valid)

    if math.isclose(
        minimum,
        maximum,
    ):
        midpoint = (
            low + high
        ) / 2.0

        return [
            midpoint
            if value is not None
            else None
            for value in values
        ]

    output = []

    for value in values:

        if value is None:
            output.append(None)
            continue

        score = (
            low
            + (
                (value - minimum)
                / (maximum - minimum)
            )
            * (high - low)
        )

        output.append(score)

    return output


def _inverse_normalize(
    values,
    low=10.0,
    high=100.0,
):
    """
    Normalize where lower raw values are better.
    """

    normalized = _min_max_normalize(
        values,
        low,
        high,
    )

    return [
        None
        if value is None
        else high + low - value
        for value in normalized
    ]


def _weighted_average(
    values,
    weights,
):
    """
    Weighted average over available values only.
    """

    usable = []

    for key, weight in weights.items():

        value = values.get(key)

        if value is None:
            continue

        usable.append(
            (
                value,
                weight,
            )
        )

    if not usable:
        return None

    total_weight = sum(
        weight
        for _, weight in usable
    )

    if total_weight <= 0:
        return None

    return sum(
        value
        * (
            weight
            / total_weight
        )
        for value, weight in usable
    )


def _haversine_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Great-circle distance between two coordinates.
    """

    radius = 6371.0

    phi1 = math.radians(
        lat1
    )

    phi2 = math.radians(
        lat2
    )

    d_phi = math.radians(
        lat2 - lat1
    )

    d_lambda = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(d_lambda / 2) ** 2
    )

    return (
        2
        * radius
        * math.asin(
            math.sqrt(a)
        )
    )


# ============================================================
# EXISTING STORE CONFIGURATION
# ============================================================

def _find_existing_store_collection():
    """
    Find the configured fixed existing-store collection.

    The function supports a few sensible names so the engine
    remains compatible with the existing project configuration.
    """

    possible_names = [
        "EXISTING_STORES",
        "FIXED_STORES",
        "EXISTING_LOCATIONS",
        "STORES",
    ]

    for name in possible_names:

        value = getattr(
            config,
            name,
            None,
        )

        if value:
            return value

    raise AttributeError(
        "No existing-store collection was found in "
        "src.config.py.\n\n"
        "Expected one of:\n"
        "EXISTING_STORES\n"
        "FIXED_STORES\n"
        "EXISTING_LOCATIONS\n"
        "STORES"
    )


def _normalize_existing_stores():
    """
    Convert existing-store configuration into a common format.

    Expected fields are normally:
        id
        name
        lat
        lon

    Optional performance fields supported:
        performance_index
        performance
        annual_sales
        sales
        store_performance

    If performance is unavailable, all fixed stores receive
    equal weight rather than fabricated sales values.
    """

    raw_stores = (
        _find_existing_store_collection()
    )

    stores = []

    for index, store in enumerate(
        raw_stores
    ):

        store_id = (
            store.get("id")
            or store.get("store_id")
            or f"existing_{index + 1}"
        )

        name = (
            store.get("name")
            or store.get("store_name")
            or str(store_id)
        )

        lat = (
            store.get("lat")
            if store.get("lat") is not None
            else store.get("latitude")
        )

        lon = (
            store.get("lon")
            if store.get("lon") is not None
            else store.get("longitude")
        )

        lat = _to_float(lat)
        lon = _to_float(lon)

        if lat is None or lon is None:
            raise ValueError(
                f"Existing store '{name}' is missing "
                "valid latitude/longitude."
            )

        performance = None

        performance_keys = [
            "performance_index",
            "performance",
            "annual_sales",
            "sales",
            "store_performance",
        ]

        for key in performance_keys:

            candidate_value = _to_float(
                store.get(key)
            )

            if candidate_value is not None:
                performance = (
                    candidate_value
                )
                break

        stores.append(
            {
                "id": store_id,
                "name": name,
                "lat": lat,
                "lon": lon,
                "performance": performance,
            }
        )

    if not stores:
        raise ValueError(
            "Existing-store configuration is empty."
        )

    # --------------------------------------------------------
    # Performance weights
    #
    # When performance is unavailable:
    #     every store gets equal weight.
    #
    # When performance is available:
    #     higher-performing stores contribute more to
    #     cannibalisation pressure.
    # --------------------------------------------------------

    has_performance = all(
        store["performance"] is not None
        for store in stores
    )

    if has_performance:

        for store in stores:

            store[
                "performance_weight"
            ] = store["performance"]

        performance_values = [
            store["performance_weight"]
            for store in stores
        ]

        total = sum(
            performance_values
        )

        if total <= 0:
            has_performance = False

        else:

            for store in stores:

                store[
                    "performance_weight"
                ] = (
                    store[
                        "performance_weight"
                    ]
                    / total
                )

    if not has_performance:

        equal_weight = (
            DEFAULT_EXISTING_STORE_WEIGHT
            / len(stores)
        )

        for store in stores:

            store[
                "performance_weight"
            ] = equal_weight

    return stores


# ============================================================
# CANNIBALISATION CALCULATION
# ============================================================

def calculate_cannibalisation(
    candidate_rows,
    existing_stores,
):
    """
    Calculate network-overlap pressure for each candidate.

    For every candidate / existing-store pair:

        Overlap
        =
        exp(
            - distance / decay_parameter
        )

    Candidate-level pressure:

        Cannibalisation Pressure
        =
        Σ(
            Existing Store Weight
            ×
            Distance Overlap
        )

    This is a relative overlap indicator.

    It is NOT a claim that a specific percentage of sales
    will be transferred.
    """

    results = []

    for candidate in candidate_rows:

        candidate_lat = _to_float(
            candidate.get("latitude")
        )

        candidate_lon = _to_float(
            candidate.get("longitude")
        )

        if (
            candidate_lat is None
            or candidate_lon is None
        ):
            raise ValueError(
                f"Candidate {candidate.get('area')} "
                "is missing coordinates."
            )

        store_details = []

        weighted_pressure = 0.0

        nearest_store = None
        nearest_distance = None

        strongest_overlap_store = None
        strongest_overlap = None

        for store in existing_stores:

            distance = _haversine_km(
                candidate_lat,
                candidate_lon,
                store["lat"],
                store["lon"],
            )

            overlap = math.exp(
                -distance
                / DISTANCE_DECAY_KM
            )

            weighted_overlap = (
                store[
                    "performance_weight"
                ]
                * overlap
            )

            weighted_pressure += (
                weighted_overlap
            )

            store_details.append(
                {
                    "existing_store_id":
                        store["id"],

                    "existing_store_name":
                        store["name"],

                    "distance_km":
                        round(
                            distance,
                            3,
                        ),

                    "overlap_score":
                        round(
                            overlap,
                            5,
                        ),

                    "performance_weight":
                        round(
                            store[
                                "performance_weight"
                            ],
                            5,
                        ),

                    "weighted_overlap":
                        round(
                            weighted_overlap,
                            5,
                        ),
                }
            )

            if (
                nearest_distance is None
                or distance < nearest_distance
            ):

                nearest_distance = distance
                nearest_store = store

            if (
                strongest_overlap is None
                or overlap > strongest_overlap
            ):

                strongest_overlap = overlap
                strongest_overlap_store = store

        results.append(
            {
                "area_id": candidate.get(
                    "area_id"
                ),

                "area": candidate.get(
                    "area"
                ),

                "existing_store_count":
                    len(existing_stores),

                "nearest_existing_store":
                    (
                        nearest_store["name"]
                        if nearest_store
                        else None
                    ),

                "nearest_existing_store_distance_km":
                    (
                        round(
                            nearest_distance,
                            3,
                        )
                        if nearest_distance
                        is not None
                        else None
                    ),

                "strongest_overlap_store":
                    (
                        strongest_overlap_store[
                            "name"
                        ]
                        if strongest_overlap_store
                        else None
                    ),

                "strongest_overlap_score":
                    (
                        round(
                            strongest_overlap,
                            5,
                        )
                        if strongest_overlap
                        is not None
                        else None
                    ),

                "cannibalisation_pressure":
                    round(
                        weighted_pressure,
                        5,
                    ),

                # Keep pair-level information as a compact
                # JSON string so the CSV remains one row/candidate.
                "existing_store_overlap_detail":
                    str(
                        store_details
                    ),
            }
        )

    return results


# ============================================================
# NETWORK IMPACT ENGINE
# ============================================================

def calculate_network_impact(
    candidate_rows,
    capture_rows,
    existing_stores,
):
    """
    Combine capturable demand with network overlap.

    Outputs:

      1. Cannibalisation Pressure
      2. Cannibalisation Attractiveness
      3. Network Impact Score
      4. Incremental Network Value Proxy

    Because actual historical network sales are not available,
    the final value metric is explicitly a PROXY index.

    Production upgrade:
        New Store Value
        -
        Transferred Existing-Store Value
        =
        Incremental Network Value
    """

    capture_lookup = {
        row.get("area_id"): row
        for row in capture_rows
    }

    cannibalisation_rows = (
        calculate_cannibalisation(
            candidate_rows,
            existing_stores,
        )
    )

    pressure_values = [
        row[
            "cannibalisation_pressure"
        ]
        for row in cannibalisation_rows
    ]

    pressure_scores = _min_max_normalize(
        pressure_values
    )

    protection_scores = _inverse_normalize(
        pressure_values
    )

    results = []

    for index, candidate in enumerate(
        candidate_rows
    ):

        area_id = candidate.get(
            "area_id"
        )

        area = candidate.get(
            "area"
        )

        capture = capture_lookup.get(
            area_id
        )

        if capture is None:
            raise ValueError(
                f"Missing Demand Capture result "
                f"for {area}."
            )

        captured_demand_score = _to_float(
            capture.get(
                "captured_demand_score"
            )
        )

        if captured_demand_score is None:
            raise ValueError(
                f"Missing captured demand score "
                f"for {area}."
            )

        cannibalisation = (
            cannibalisation_rows[index]
        )

        cannibalisation_pressure = (
            cannibalisation[
                "cannibalisation_pressure"
            ]
        )

        cannibalisation_score = (
            pressure_scores[index]
        )

        protection_score = (
            protection_scores[index]
        )

        network_impact_score = (
            _weighted_average(
                {
                    "captured_demand":
                        captured_demand_score,

                    "cannibalisation_protection":
                        protection_score,
                },
                {
                    "captured_demand":
                        CAPTURE_WEIGHT,

                    "cannibalisation_protection":
                        CANNIBALISATION_PROTECTION_WEIGHT,
                },
            )
        )

        # ----------------------------------------------------
        # A relative network value proxy.
        #
        # Higher captured demand increases opportunity.
        # Higher cannibalisation decreases network value.
        #
        # This remains an index, not rupee profit.
        # ----------------------------------------------------

        if network_impact_score is None:

            incremental_value_proxy = None

        else:

            incremental_value_proxy = (
                network_impact_score
            )

        results.append(
            {
                "area_id": area_id,
                "area": area,

                # ----------------------------------------
                # Demand
                # ----------------------------------------

                "captured_demand_score":
                    round(
                        captured_demand_score,
                        2,
                    ),

                # ----------------------------------------
                # Existing network
                # ----------------------------------------

                "existing_store_count":
                    cannibalisation[
                        "existing_store_count"
                    ],

                "nearest_existing_store":
                    cannibalisation[
                        "nearest_existing_store"
                    ],

                "nearest_existing_store_distance_km":
                    cannibalisation[
                        "nearest_existing_store_distance_km"
                    ],

                "strongest_overlap_store":
                    cannibalisation[
                        "strongest_overlap_store"
                    ],

                "strongest_overlap_score":
                    cannibalisation[
                        "strongest_overlap_score"
                    ],

                # ----------------------------------------
                # Cannibalisation
                # ----------------------------------------

                "cannibalisation_pressure":
                    round(
                        cannibalisation_pressure,
                        5,
                    ),

                "cannibalisation_score":
                    (
                        round(
                            cannibalisation_score,
                            2,
                        )
                        if cannibalisation_score
                        is not None
                        else None
                    ),

                "cannibalisation_protection_score":
                    (
                        round(
                            protection_score,
                            2,
                        )
                        if protection_score
                        is not None
                        else None
                    ),

                # ----------------------------------------
                # Network-level result
                # ----------------------------------------

                "network_impact_score":
                    (
                        round(
                            network_impact_score,
                            2,
                        )
                        if network_impact_score
                        is not None
                        else None
                    ),

                "incremental_network_value_proxy":
                    (
                        round(
                            incremental_value_proxy,
                            2,
                        )
                        if incremental_value_proxy
                        is not None
                        else None
                    ),

                "network_model_type":
                    "distance_decay_overlap",

                "network_value_status":
                    "prototype_proxy",
            }
        )

    return results


# ============================================================
# CSV LOADING
# ============================================================

def load_csv(path):
    """
    Generic CSV loader.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(
            file
        )

        rows = list(reader)

    if not rows:
        raise ValueError(
            f"CSV file is empty: {path}"
        )

    return rows


# ============================================================
# CSV OUTPUT
# ============================================================

def write_network_impact_csv(
    rows,
    output_path=OUTPUT_FILE,
):
    """
    Write network-impact results.
    """

    if not rows:
        raise ValueError(
            "Cannot write empty network-impact output."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
    )

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
# END-TO-END PIPELINE
# ============================================================

def run_network_impact(
    feature_file=FEATURE_FILE,
    capture_file=CAPTURE_FILE,
    output_file=OUTPUT_FILE,
):
    """
    Complete Network Impact pipeline.
    """

    candidate_rows = load_csv(
        feature_file
    )

    capture_rows = load_csv(
        capture_file
    )

    existing_stores = (
        _normalize_existing_stores()
    )

    results = calculate_network_impact(
        candidate_rows,
        capture_rows,
        existing_stores,
    )

    output_path = (
        write_network_impact_csv(
            results,
            output_file,
        )
    )

    return {
        "results": results,
        "output_path": output_path,
        "existing_stores":
            existing_stores,
    }


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    result = run_network_impact()

    print(
        "Network Impact Engine completed.\n"
    )

    print(
        f"Fixed existing stores: "
        f"{len(result['existing_stores'])}\n"
    )

    for store in result[
        "existing_stores"
    ]:

        print(
            f"  - {store['name']}"
        )

    print(
        f"\nOutput file:\n"
        f"{result['output_path']}\n"
    )

    for row in result["results"]:

        print(
            f"{row['area']}: "
            f"captured_demand="
            f"{row['captured_demand_score']}, "
            f"nearest_existing="
            f"{row['nearest_existing_store']} "
            f"({row['nearest_existing_store_distance_km']} km), "
            f"cannibalisation_pressure="
            f"{row['cannibalisation_pressure']}, "
            f"network_impact="
            f"{row['network_impact_score']}"
        )
