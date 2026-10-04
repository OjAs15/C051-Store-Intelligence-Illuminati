import csv
import math
from pathlib import Path


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

DEMAND_FILE = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "processed"
    / "demand_prediction.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "processed"
    / "demand_capture.csv"
)


# ============================================================
# CAPTURE MODEL CONFIGURATION
# ============================================================

# Accessibility:
#   60% transit density
#   40% proximity to nearest transit
#
# Competition:
#   70% distance-decayed competitive pressure
#   30% competitor count
#
# These weights are deliberately simple and explainable for
# the proof-of-concept. They can later be calibrated using
# actual retailer performance and customer catchment data.

ACCESSIBILITY_WEIGHTS = {
    "transit_density_per_sqkm": 0.60,
    "nearest_transit_proximity": 0.40,
}

COMPETITION_WEIGHTS = {
    "competitor_pressure": 0.70,
    "competitor_count": 0.30,
}


# Factor ranges:
#
# Accessibility:
#   0.80 = weaker accessibility
#   1.20 = stronger accessibility
#
# Competition:
#   0.80 = heavier competitive pressure
#   1.10 = lighter competitive pressure
#
# These are adjustment multipliers, not probabilities.

ACCESSIBILITY_MIN_FACTOR = 0.80
ACCESSIBILITY_MAX_FACTOR = 1.20

COMPETITION_MIN_FACTOR = 0.80
COMPETITION_MAX_FACTOR = 1.10


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
    low=0.0,
    high=100.0,
):
    """
    Min-max normalize values to [low, high].

    Identical valid values receive the midpoint because the
    feature does not differentiate candidates.
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
        ) / 2

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

        normalized = (
            low
            + (
                (value - minimum)
                / (maximum - minimum)
            )
            * (high - low)
        )

        output.append(normalized)

    return output


def _inverse_normalize(
    values,
    low=0.0,
    high=100.0,
):
    """
    Normalize where LOWER raw values are better.

    Example:
      lower nearest-transit distance
      lower competitor count
      lower competitor pressure
    """

    normalized = _min_max_normalize(
        values,
        low,
        high,
    )

    return [
        (
            None
            if value is None
            else high - value + low
        )
        for value in normalized
    ]


def _weighted_average(
    values,
    weights,
):
    """
    Weighted average that ignores missing inputs and
    renormalizes the available weights.
    """

    usable = []

    for feature, weight in weights.items():

        value = values.get(
            feature
        )

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


def _scale_factor(
    score,
    minimum_factor,
    maximum_factor,
):
    """
    Convert a 0-100 component score into an adjustment factor.
    """

    if score is None:
        return None

    return (
        minimum_factor
        + (
            score / 100.0
        )
        * (
            maximum_factor
            - minimum_factor
        )
    )


# ============================================================
# INPUT LOADING
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
# ACCESSIBILITY COMPONENT
# ============================================================

def calculate_accessibility_score(
    feature_rows
):
    """
    Calculate relative accessibility for each candidate.

    Inputs:
      - transit density
      - nearest transit distance

    Higher transit density is better.
    Lower nearest-transit distance is better.
    """

    density_values = [
        _to_float(
            row.get(
                "transit_density_per_sqkm"
            )
        )
        for row in feature_rows
    ]

    nearest_values = [
        _to_float(
            row.get(
                "nearest_transit_distance_km"
            )
        )
        for row in feature_rows
    ]

    density_scores = _min_max_normalize(
        density_values
    )

    proximity_scores = _inverse_normalize(
        nearest_values
    )

    results = []

    for index, row in enumerate(
        feature_rows
    ):

        component_values = {
            "transit_density_per_sqkm":
                density_scores[index],

            "nearest_transit_proximity":
                proximity_scores[index],
        }

        score = _weighted_average(
            component_values,
            ACCESSIBILITY_WEIGHTS,
        )

        factor = _scale_factor(
            score,
            ACCESSIBILITY_MIN_FACTOR,
            ACCESSIBILITY_MAX_FACTOR,
        )

        results.append(
            {
                "area_id": row.get(
                    "area_id"
                ),
                "area": row.get(
                    "area"
                ),

                "transit_density_score":
                    None
                    if density_scores[index]
                    is None
                    else round(
                        density_scores[index],
                        2,
                    ),

                "transit_proximity_score":
                    None
                    if proximity_scores[index]
                    is None
                    else round(
                        proximity_scores[index],
                        2,
                    ),

                "accessibility_score":
                    None
                    if score is None
                    else round(
                        score,
                        2,
                    ),

                "accessibility_factor":
                    None
                    if factor is None
                    else round(
                        factor,
                        4,
                    ),
            }
        )

    return results


# ============================================================
# COMPETITION COMPONENT
# ============================================================

def calculate_competition_score(
    feature_rows
):
    """
    Calculate relative competitive attractiveness.

    IMPORTANT:
    Lower competition = higher score.

    Inputs:
      - distance-decayed competitor pressure
      - competitor count
    """

    pressure_values = [
        _to_float(
            row.get(
                "competitor_pressure"
            )
        )
        for row in feature_rows
    ]

    count_values = [
        _to_float(
            row.get(
                "competitor_count"
            )
        )
        for row in feature_rows
    ]

    pressure_scores = _inverse_normalize(
        pressure_values
    )

    count_scores = _inverse_normalize(
        count_values
    )

    results = []

    for index, row in enumerate(
        feature_rows
    ):

        component_values = {
            "competitor_pressure":
                pressure_scores[index],

            "competitor_count":
                count_scores[index],
        }

        score = _weighted_average(
            component_values,
            COMPETITION_WEIGHTS,
        )

        factor = _scale_factor(
            score,
            COMPETITION_MIN_FACTOR,
            COMPETITION_MAX_FACTOR,
        )

        results.append(
            {
                "area_id": row.get(
                    "area_id"
                ),
                "area": row.get(
                    "area"
                ),

                "competitive_pressure_score":
                    None
                    if pressure_scores[index]
                    is None
                    else round(
                        pressure_scores[index],
                        2,
                    ),

                "competitive_count_score":
                    None
                    if count_scores[index]
                    is None
                    else round(
                        count_scores[index],
                        2,
                    ),

                "competition_score":
                    None
                    if score is None
                    else round(
                        score,
                        2,
                    ),

                "competition_factor":
                    None
                    if factor is None
                    else round(
                        factor,
                        4,
                    ),
            }
        )

    return results


# ============================================================
# DEMAND CAPTURE ENGINE
# ============================================================

def calculate_demand_capture(
    feature_rows,
    demand_rows,
):
    """
    Combine ML predicted demand with accessibility and
    competition adjustments.

    Core logic:

        Capturable Demand
            =
        Predicted Demand
            ×
        Accessibility Factor
            ×
        Competition Factor

    The output remains a relative demand index.
    It is NOT revenue and NOT a probability of purchase.
    """

    accessibility = calculate_accessibility_score(
        feature_rows
    )

    competition = calculate_competition_score(
        feature_rows
    )

    demand_lookup = {
        row.get("area_id"): row
        for row in demand_rows
    }

    accessibility_lookup = {
        row.get("area_id"): row
        for row in accessibility
    }

    competition_lookup = {
        row.get("area_id"): row
        for row in competition
    }

    results = []

    for feature_row in feature_rows:

        area_id = feature_row.get(
            "area_id"
        )

        area = feature_row.get(
            "area"
        )

        demand = demand_lookup.get(
            area_id
        )

        access = accessibility_lookup.get(
            area_id
        )

        comp = competition_lookup.get(
            area_id
        )

        if demand is None:
            raise ValueError(
                f"Missing demand prediction "
                f"for {area}."
            )

        if access is None:
            raise ValueError(
                f"Missing accessibility "
                f"result for {area}."
            )

        if comp is None:
            raise ValueError(
                f"Missing competition "
                f"result for {area}."
            )

        predicted_demand = _to_float(
            demand.get(
                "predicted_demand_index"
            )
        )

        accessibility_factor = _to_float(
            access.get(
                "accessibility_factor"
            )
        )

        competition_factor = _to_float(
            comp.get(
                "competition_factor"
            )
        )

        if predicted_demand is None:
            raise ValueError(
                f"Missing predicted demand "
                f"for {area}."
            )

        if accessibility_factor is None:
            raise ValueError(
                f"Missing accessibility factor "
                f"for {area}."
            )

        if competition_factor is None:
            raise ValueError(
                f"Missing competition factor "
                f"for {area}."
            )

        captured_demand = (
            predicted_demand
            * accessibility_factor
            * competition_factor
        )

        results.append(
            {
                "area_id": area_id,
                "area": area,

                # ----------------------------------------
                # Demand input
                # ----------------------------------------

                "predicted_demand_index":
                    round(
                        predicted_demand,
                        2,
                    ),

                # ----------------------------------------
                # Accessibility
                # ----------------------------------------

                "accessibility_score":
                    access.get(
                        "accessibility_score"
                    ),

                "accessibility_factor":
                    round(
                        accessibility_factor,
                        4,
                    ),

                # ----------------------------------------
                # Competition
                # ----------------------------------------

                "competition_score":
                    comp.get(
                        "competition_score"
                    ),

                "competition_factor":
                    round(
                        competition_factor,
                        4,
                    ),

                # ----------------------------------------
                # Final captured demand
                # ----------------------------------------

                "captured_demand_index":
                    round(
                        captured_demand,
                        2,
                    ),
            }
        )

    # --------------------------------------------------------
    # Normalize final captured demand to relative 10-100
    # --------------------------------------------------------

    captured_values = [
        row["captured_demand_index"]
        for row in results
    ]

    captured_scores = _min_max_normalize(
        captured_values,
        low=10.0,
        high=100.0,
    )

    for index, row in enumerate(
        results
    ):
        row[
            "captured_demand_score"
        ] = round(
            captured_scores[index],
            2,
        )

    return results


# ============================================================
# OUTPUT WRITER
# ============================================================

def write_capture_csv(
    rows,
    output_path=OUTPUT_FILE,
):
    """
    Write demand-capture output.
    """

    if not rows:
        raise ValueError(
            "Cannot write empty capture output."
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

def run_demand_capture(
    feature_file=FEATURE_FILE,
    demand_file=DEMAND_FILE,
    output_file=OUTPUT_FILE,
):
    """
    Complete Demand Capture pipeline.
    """

    feature_rows = load_csv(
        feature_file
    )

    demand_rows = load_csv(
        demand_file
    )

    results = calculate_demand_capture(
        feature_rows,
        demand_rows,
    )

    output_path = write_capture_csv(
        results,
        output_file,
    )

    return {
        "results": results,
        "output_path": output_path,
    }


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    result = run_demand_capture()

    print(
        "Demand Capture Engine completed.\n"
    )

    print(
        f"Output file:\n"
        f"{result['output_path']}\n"
    )

    for row in result["results"]:

        print(
            f"{row['area']}: "
            f"predicted_demand="
            f"{row['predicted_demand_index']}, "
            f"accessibility_factor="
            f"{row['accessibility_factor']}, "
            f"competition_factor="
            f"{row['competition_factor']}, "
            f"captured_demand="
            f"{row['captured_demand_index']}, "
            f"capture_score="
            f"{row['captured_demand_score']}"
        )
