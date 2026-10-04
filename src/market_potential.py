import csv
import math
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "v2"
    / "processed"
    / "candidate_features.csv"
)

OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "v2"
    / "processed"
    / "market_potential.csv"
)


# These weights represent the relative importance of the
# underlying market opportunity.
#
# Population = customer volume
# Purchasing power = spending capacity
# Households = number of potential household demand units
#
# The weights sum to 1.0.

WEIGHTS = {
    "population_density_per_sqkm": 0.40,
    "purchasing_power_proxy": 0.40,
    "household_density_per_sqkm": 0.20,
}


# ============================================================
# HELPERS
# ============================================================

def _to_float(value):
    """
    Safely convert a value into float.
    """
    if value is None:
        return None

    if value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _min_max_normalize(values):
    """
    Convert values into a 10-100 scale.

    Why 10 instead of 0?
    A zero in a geometric index would mathematically collapse
    the entire score. A 10-100 scale preserves relative
    differences while keeping the geometric calculation stable.

    When all valid values are identical, return 55 for each
    valid observation because that feature provides no
    differentiation between candidates.
    """

    valid_values = [
        value for value in values
        if value is not None
    ]

    if not valid_values:
        return [None for _ in values]

    minimum = min(valid_values)
    maximum = max(valid_values)

    if math.isclose(minimum, maximum):
        return [
            55.0 if value is not None else None
            for value in values
        ]

    normalized = []

    for value in values:

        if value is None:
            normalized.append(None)
            continue

        score = (
            10
            + 90
            * (
                (value - minimum)
                / (maximum - minimum)
            )
        )

        normalized.append(score)

    return normalized


def _weighted_geometric(values, weights):
    """
    Weighted geometric mean.

    Missing features are excluded and the remaining weights
    are re-normalized.

    This avoids inventing a value when a public data field is
    unavailable.
    """

    usable = []

    for feature, weight in weights.items():

        value = values.get(feature)

        if value is None:
            continue

        usable.append(
            (value, weight)
        )

    if not usable:
        return None

    total_weight = sum(
        weight
        for _, weight in usable
    )

    if total_weight <= 0:
        return None

    weighted_log_sum = 0.0

    for value, weight in usable:

        # Defensive floor because logarithm cannot accept 0.
        safe_value = max(value, 0.0001)

        weighted_log_sum += (
            (weight / total_weight)
            * math.log(safe_value)
        )

    return math.exp(weighted_log_sum)


# ============================================================
# CSV INPUT
# ============================================================

def load_feature_table(path=FEATURE_FILE):
    """
    Load candidate feature table from CSV.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}\n"
            "Run feature engineering first."
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    if not rows:
        raise ValueError(
            "Feature file exists but contains no rows."
        )

    return rows


# ============================================================
# MARKET POTENTIAL ENGINE
# ============================================================

def calculate_market_potential(rows):
    """
    Calculate a relative Market Potential score for every
    candidate.

    Logic:

        Raw market signals
              ↓
        Min-max normalization
              ↓
        Weighted geometric index
              ↓
        Market Potential Score

    This measures underlying market opportunity.

    It deliberately does NOT consider:
      - competition
      - accessibility
      - rent
      - cannibalisation
      - predicted store sales

    Those belong to downstream engines.
    """

    feature_columns = list(
        WEIGHTS.keys()
    )

    # --------------------------------------------------------
    # Extract raw values
    # --------------------------------------------------------

    raw_columns = {
        column: [
            _to_float(row.get(column))
            for row in rows
        ]
        for column in feature_columns
    }

    # --------------------------------------------------------
    # Normalize each variable
    # --------------------------------------------------------

    normalized_columns = {
        column: _min_max_normalize(
            raw_columns[column]
        )
        for column in feature_columns
    }

    output = []

    for index, row in enumerate(rows):

        normalized_values = {}

        for column in feature_columns:
            normalized_values[column] = (
                normalized_columns[column][index]
            )

        market_score = _weighted_geometric(
            normalized_values,
            WEIGHTS,
        )

        usable_features = sum(
            value is not None
            for value in normalized_values.values()
        )

        total_features = len(
            normalized_values
        )

        if usable_features == total_features:
            data_quality = "complete"

        elif usable_features > 0:
            data_quality = "partial"

        else:
            data_quality = "insufficient"

        output.append(
            {
                "area_id": row.get("area_id"),
                "area": row.get("area"),

                # ----------------------------------------
                # Raw inputs
                # ----------------------------------------

                "population_density_per_sqkm":
                    _to_float(
                        row.get(
                            "population_density_per_sqkm"
                        )
                    ),

                "purchasing_power_proxy":
                    _to_float(
                        row.get(
                            "purchasing_power_proxy"
                        )
                    ),

                "household_density_per_sqkm":
                    _to_float(
                        row.get(
                            "household_density_per_sqkm"
                        )
                    ),

                # ----------------------------------------
                # Normalized inputs
                # ----------------------------------------

                "population_market_index":
                    normalized_values[
                        "population_density_per_sqkm"
                    ],

                "purchasing_power_market_index":
                    normalized_values[
                        "purchasing_power_proxy"
                    ],

                "household_market_index":
                    normalized_values[
                        "household_density_per_sqkm"
                    ],

                # ----------------------------------------
                # Final engine output
                # ----------------------------------------

                "market_potential_score":
                    round(
                        market_score,
                        2,
                    )
                    if market_score is not None
                    else None,

                "market_data_quality":
                    data_quality,
            }
        )

    return output


# ============================================================
# CSV OUTPUT
# ============================================================

def write_market_potential_csv(
    rows,
    output_path=OUTPUT_FILE,
):
    """
    Write Market Potential results to CSV.
    """

    if not rows:
        raise ValueError(
            "Cannot write Market Potential CSV: "
            "no rows supplied."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
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
# END-TO-END ENGINE
# ============================================================

def run_market_potential(
    feature_file=FEATURE_FILE,
    output_file=OUTPUT_FILE,
):
    """
    Load features, calculate Market Potential and save output.
    """

    rows = load_feature_table(
        feature_file
    )

    results = calculate_market_potential(
        rows
    )

    output_path = write_market_potential_csv(
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

    result = run_market_potential()

    print(
        f"Market Potential results written to:\n"
        f"{result['output_path']}\n"
    )

    for row in result["results"]:

        print(
            f"{row['area']}: "
            f"Market Potential = "
            f"{row['market_potential_score']}"
        )
