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
    / "store_economics.csv"
)


# ============================================================
# ECONOMIC MODEL CONFIGURATION
# ============================================================

# When rent is available:
#
# 60% = demand attractiveness
# 40% = rent attractiveness
#
# This intentionally does NOT claim profitability because
# we do not yet have:
#   - COGS
#   - staffing cost
#   - utilities
#   - capex
#   - maintenance
#   - taxes
#
# Therefore the current output is an ECONOMIC ATTRACTIVENESS
# signal, not profit or ROI.

DEMAND_WEIGHT = 0.60
RENT_WEIGHT = 0.40


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
    Normalize values to a specified range.

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

    result = []

    for value in values:

        if value is None:
            result.append(None)
            continue

        score = (
            low
            + (
                (value - minimum)
                / (maximum - minimum)
            )
            * (high - low)
        )

        result.append(score)

    return result


def _inverse_normalize(
    values,
    low=0.0,
    high=100.0,
):
    """
    Normalize where a lower raw value is more attractive.

    Used for rent:
        lower rent = better economic attractiveness.
    """

    normalized = _min_max_normalize(
        values,
        low,
        high,
    )

    return [
        None
        if value is None
        else high - value + low
        for value in normalized
    ]


def _weighted_average(
    values,
    weights,
):
    """
    Weighted average using only available values.

    Missing inputs cause their weights to be removed and the
    remaining weights to be renormalized.
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


# ============================================================
# INPUT LOADING
# ============================================================

def load_csv(path):
    """
    Load a CSV file.
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
            f"CSV file contains no rows: {path}"
        )

    return rows


# ============================================================
# STORE ECONOMICS ENGINE
# ============================================================

def calculate_store_economics(
    feature_rows,
    capture_rows,
):
    """
    Calculate relative economic attractiveness.

    Current V2 logic:

        Demand attractiveness
              +
        Rent attractiveness
              ↓
        Economic Attractiveness Score

    If rent is available:

        Economics Score
        =
        60% Demand
        +
        40% Rent Attractiveness

    If rent is unavailable:

        Economics Score
        =
        Demand score only

    but the result is explicitly marked as PARTIAL.

    This is intentional because we must not fabricate rent
    or claim financial returns without sufficient cost data.
    """

    capture_lookup = {
        row.get("area_id"): row
        for row in capture_rows
    }

    results = []

    for row in feature_rows:

        area_id = row.get(
            "area_id"
        )

        area = row.get(
            "area"
        )

        capture = capture_lookup.get(
            area_id
        )

        if capture is None:
            raise ValueError(
                f"Missing Demand Capture "
                f"result for {area}."
            )

        captured_demand_score = _to_float(
            capture.get(
                "captured_demand_score"
            )
        )

        annual_rent = _to_float(
            row.get(
                "annual_rent"
            )
        )

        monthly_rent = _to_float(
            row.get(
                "monthly_rent"
            )
        )

        # ----------------------------------------------------
        # If monthly rent exists but annual rent doesn't,
        # derive annual rent mechanically.
        # ----------------------------------------------------

        if (
            annual_rent is None
            and monthly_rent is not None
        ):
            annual_rent = (
                monthly_rent * 12
            )

        results.append(
            {
                "area_id": area_id,
                "area": area,

                "captured_demand_score":
                    captured_demand_score,

                "monthly_rent":
                    monthly_rent,

                "annual_rent":
                    annual_rent,

                "rent_status":
                    row.get(
                        "rent_status",
                        "unknown",
                    ),
            }
        )

    # --------------------------------------------------------
    # Rent attractiveness
    # --------------------------------------------------------

    annual_rents = [
        row["annual_rent"]
        for row in results
    ]

    rent_scores = _inverse_normalize(
        annual_rents
    )

    # --------------------------------------------------------
    # Final economics score
    # --------------------------------------------------------

    for index, row in enumerate(
        results
    ):

        demand_score = row[
            "captured_demand_score"
        ]

        rent_score = rent_scores[index]

        components = {
            "demand": demand_score,
            "rent": rent_score,
        }

        weights = {
            "demand": DEMAND_WEIGHT,
            "rent": RENT_WEIGHT,
        }

        economics_score = _weighted_average(
            components,
            weights,
        )

        if rent_score is None:

            data_quality = "partial"

            explanation = (
                "Rent unavailable; score uses "
                "captured demand only."
            )

            rent_weight_used = 0.0

        else:

            data_quality = "complete"

            explanation = (
                "Score combines captured demand "
                "with relative rent attractiveness."
            )

            rent_weight_used = RENT_WEIGHT

        row[
            "rent_attractiveness_score"
        ] = (
            None
            if rent_score is None
            else round(
                rent_score,
                2,
            )
        )

        row[
            "economic_attractiveness_score"
        ] = (
            None
            if economics_score is None
            else round(
                economics_score,
                2,
            )
        )

        row[
            "demand_weight_used"
        ] = round(
            (
                DEMAND_WEIGHT
                if demand_score is not None
                else 0.0
            ),
            2,
        )

        row[
            "rent_weight_used"
        ] = round(
            rent_weight_used,
            2,
        )

        row[
            "economics_data_quality"
        ] = data_quality

        row[
            "economics_explanation"
        ] = explanation

    return results


# ============================================================
# OUTPUT WRITER
# ============================================================

def write_economics_csv(
    rows,
    output_path=OUTPUT_FILE,
):
    """
    Write economics results.
    """

    if not rows:
        raise ValueError(
            "Cannot write empty economics output."
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

def run_store_economics(
    feature_file=FEATURE_FILE,
    capture_file=CAPTURE_FILE,
    output_file=OUTPUT_FILE,
):
    """
    Complete Store Economics pipeline.
    """

    feature_rows = load_csv(
        feature_file
    )

    capture_rows = load_csv(
        capture_file
    )

    results = calculate_store_economics(
        feature_rows,
        capture_rows,
    )

    output_path = write_economics_csv(
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

    result = run_store_economics()

    print(
        "Store Economics Engine completed.\n"
    )

    print(
        f"Output file:\n"
        f"{result['output_path']}\n"
    )

    for row in result["results"]:

        print(
            f"{row['area']}: "
            f"annual_rent="
            f"{row['annual_rent']}, "
            f"rent_score="
            f"{row['rent_attractiveness_score']}, "
            f"economics_score="
            f"{row['economic_attractiveness_score']}, "
            f"quality="
            f"{row['economics_data_quality']}"
        )
