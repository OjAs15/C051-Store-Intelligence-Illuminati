import csv
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


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

TRAINING_DIR = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "synthetic"
)

TRAINING_FILE = (
    TRAINING_DIR
    / "demand_training_data.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "processed"
    / "demand_prediction.csv"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

FEATURE_COLUMNS = [
    "population_density_per_sqkm",
    "purchasing_power_proxy",
    "household_density_per_sqkm",
    "transit_density_per_sqkm",
]

MODEL_RANDOM_STATE = 42


# ============================================================
# SAFE HELPERS
# ============================================================

def _to_float(value):
    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _min_max_normalize(values):
    """
    Normalize a list to 10-100.

    A 10-100 range avoids zeros and keeps the output
    interpretable as a relative demand index.
    """

    valid = [
        value
        for value in values
        if value is not None
    ]

    if not valid:
        return [None for _ in values]

    minimum = min(valid)
    maximum = max(valid)

    if math.isclose(minimum, maximum):
        return [
            55.0 if value is not None else None
            for value in values
        ]

    result = []

    for value in values:

        if value is None:
            result.append(None)
            continue

        normalized = (
            10
            + 90
            * (
                (value - minimum)
                / (maximum - minimum)
            )
        )

        result.append(normalized)

    return result


# ============================================================
# INPUT LOADING
# ============================================================

def load_feature_table(path=FEATURE_FILE):
    """
    Load the candidate feature table.
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
            "Candidate feature table is empty."
        )

    return rows


# ============================================================
# SYNTHETIC TRAINING DATA
# ============================================================

def generate_synthetic_training_data(
    candidate_rows,
    n_per_candidate=250,
    random_state=MODEL_RANDOM_STATE,
):
    """
    Create synthetic store observations around the real
    candidate feature ranges.

    IMPORTANT:
    This is NOT historical Croma sales data.

    It exists only to demonstrate how the ML demand
    prediction component operates when retailer-specific
    historical sales become available.

    The synthetic target is a demand index generated from
    transparent assumptions:
      - population opportunity
      - purchasing power
      - household opportunity
      - accessibility

    The model learns that relationship rather than being
    presented as a validated real-world sales forecast.
    """

    rng = np.random.default_rng(
        random_state
    )

    base_values = {}

    for column in FEATURE_COLUMNS:

        values = [
            _to_float(row.get(column))
            for row in candidate_rows
        ]

        values = [
            value
            for value in values
            if value is not None
        ]

        if not values:
            raise ValueError(
                f"No usable values found for "
                f"training feature: {column}"
            )

        base_values[column] = values

    training_rows = []

    for candidate_index in range(
        len(candidate_rows)
    ):

        candidate = candidate_rows[
            candidate_index
        ]

        for observation in range(
            n_per_candidate
        ):

            features = {}

            # ------------------------------------------------
            # Perturb the real candidate-level public signals
            # to create synthetic store observations.
            # ------------------------------------------------

            for column in FEATURE_COLUMNS:

                base = _to_float(
                    candidate.get(column)
                )

                multiplier = rng.uniform(
                    0.70,
                    1.30,
                )

                noise = rng.normal(
                    0,
                    0.035,
                )

                value = (
                    base
                    * multiplier
                    * (1 + noise)
                )

                features[column] = max(
                    value,
                    0.0001,
                )

            # ------------------------------------------------
            # Convert each dimension to a relative 0-100
            # scale using the broader candidate-derived
            # training distribution.
            # ------------------------------------------------

            normalized_dimensions = {}

            for column in FEATURE_COLUMNS:

                base_min = min(
                    base_values[column]
                )

                base_max = max(
                    base_values[column]
                )

                value = features[column]

                if math.isclose(
                    base_min,
                    base_max,
                ):
                    dimension_score = 55.0

                else:

                    ratio = (
                        value - base_min
                    ) / (
                        base_max - base_min
                    )

                    # Allow synthetic values to exceed
                    # candidate range without exploding.
                    ratio = max(
                        0.0,
                        min(1.0, ratio),
                    )

                    dimension_score = (
                        10 + 90 * ratio
                    )

                normalized_dimensions[
                    column
                ] = dimension_score

            # ------------------------------------------------
            # Transparent synthetic demand relationship.
            #
            # Population and purchasing power dominate.
            # Households provide additional market depth.
            # Accessibility provides a smaller contribution.
            # ------------------------------------------------

            synthetic_demand = (
                0.40
                * normalized_dimensions[
                    "population_density_per_sqkm"
                ]
                + 0.35
                * normalized_dimensions[
                    "purchasing_power_proxy"
                ]
                + 0.15
                * normalized_dimensions[
                    "household_density_per_sqkm"
                ]
                + 0.10
                * normalized_dimensions[
                    "transit_density_per_sqkm"
                ]
            )

            target_noise = rng.normal(
                0,
                3.0,
            )

            synthetic_demand = max(
                10.0,
                min(
                    100.0,
                    synthetic_demand
                    + target_noise,
                ),
            )

            training_rows.append(
                {
                    "source_type": "synthetic_demo",
                    "base_area": candidate.get(
                        "area"
                    ),
                    **features,
                    "demand_index": round(
                        synthetic_demand,
                        4,
                    ),
                }
            )

    return training_rows


# ============================================================
# SAVE SYNTHETIC TRAINING DATA
# ============================================================

def write_training_data(
    rows,
    output_path=TRAINING_FILE,
):
    """
    Save the synthetic training dataset.
    """

    if not rows:
        raise ValueError(
            "Cannot write empty training dataset."
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
# TRAIN MODEL
# ============================================================

def train_demand_model(training_rows):
    """
    Train HistGradientBoostingRegressor on the synthetic
    demand training data.

    Production replacement:
    historical store-level sales should replace the
    synthetic target.
    """

    X = []

    y = []

    for row in training_rows:

        feature_vector = []

        for column in FEATURE_COLUMNS:

            value = _to_float(
                row.get(column)
            )

            if value is None:
                raise ValueError(
                    f"Missing training feature: "
                    f"{column}"
                )

            feature_vector.append(value)

        target = _to_float(
            row.get("demand_index")
        )

        if target is None:
            raise ValueError(
                "Training row missing demand_index."
            )

        X.append(feature_vector)
        y.append(target)

    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=MODEL_RANDOM_STATE,
    )

    model.fit(
        np.asarray(X),
        np.asarray(y),
    )

    return model


# ============================================================
# PREDICT CANDIDATE DEMAND
# ============================================================

def predict_candidate_demand(
    candidate_rows,
    model,
):
    """
    Predict relative demand for the three candidates.
    """

    X = []

    for row in candidate_rows:

        feature_vector = []

        for column in FEATURE_COLUMNS:

            value = _to_float(
                row.get(column)
            )

            if value is None:

                raise ValueError(
                    f"Candidate {row.get('area')} "
                    f"is missing {column}."
                )

            feature_vector.append(value)

        X.append(feature_vector)

    predictions = model.predict(
        np.asarray(X)
    )

    predictions = [
        max(
            0.0,
            float(prediction),
        )
        for prediction in predictions
    ]

    normalized_scores = _min_max_normalize(
        predictions
    )

    results = []

    for index, row in enumerate(
        candidate_rows
    ):

        results.append(
            {
                "area_id": row.get(
                    "area_id"
                ),
                "area": row.get(
                    "area"
                ),

                "predicted_demand_index": round(
                    predictions[index],
                    2,
                ),

                "predicted_demand_score": round(
                    normalized_scores[index],
                    2,
                ),

                "model_type":
                    "HistGradientBoostingRegressor",

                "training_data":
                    "synthetic_demo",

                "production_status":
                    "requires_retailer_historical_sales",
            }
        )

    return results


# ============================================================
# OUTPUT WRITER
# ============================================================

def write_prediction_csv(
    rows,
    output_path=OUTPUT_FILE,
):
    """
    Save candidate demand predictions.
    """

    if not rows:
        raise ValueError(
            "Cannot write empty demand prediction output."
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
# END-TO-END PIPELINE
# ============================================================

def run_demand_prediction(
    feature_file=FEATURE_FILE,
    training_file=TRAINING_FILE,
    output_file=OUTPUT_FILE,
):
    """
    Complete demand prediction pipeline.
    """

    candidate_rows = load_feature_table(
        feature_file
    )

    training_rows = (
        generate_synthetic_training_data(
            candidate_rows
        )
    )

    write_training_data(
        training_rows,
        training_file,
    )

    model = train_demand_model(
        training_rows
    )

    predictions = predict_candidate_demand(
        candidate_rows,
        model,
    )

    output_path = write_prediction_csv(
        predictions,
        output_file,
    )

    return {
        "training_rows": training_rows,
        "predictions": predictions,
        "training_path": training_file,
        "output_path": output_path,
        "model": model,
    }


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    result = run_demand_prediction()

    print(
        "Demand Prediction Engine completed.\n"
    )

    print(
        f"Synthetic training data: "
        f"{len(result['training_rows'])} rows"
    )

    print(
        f"Training file:\n"
        f"{result['training_path']}\n"
    )

    print(
        f"Prediction file:\n"
        f"{result['output_path']}\n"
    )

    for row in result["predictions"]:

        print(
            f"{row['area']}: "
            f"predicted_demand_index="
            f"{row['predicted_demand_index']}, "
            f"demand_score="
            f"{row['predicted_demand_score']}"
        )
