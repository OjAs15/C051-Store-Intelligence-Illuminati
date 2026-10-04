from src.demand_prediction import (
    FEATURE_FILE,
    OUTPUT_FILE,
    TRAINING_FILE,
    generate_synthetic_training_data,
    load_feature_table,
    run_demand_prediction,
    train_demand_model,
)


EXPECTED_AREAS = {
    "BKC",
    "DADAR",
    "VIKHROLI",
}


def main():

    print("Testing Demand Prediction Engine...\n")

    # --------------------------------------------------------
    # Load candidate features
    # --------------------------------------------------------

    candidate_rows = load_feature_table(
        FEATURE_FILE
    )

    assert len(candidate_rows) == 3, (
        f"Expected 3 candidates, "
        f"got {len(candidate_rows)}"
    )

    returned_areas = {
        row["area"].strip().upper()
        for row in candidate_rows
    }

    assert returned_areas == EXPECTED_AREAS, (
        f"Expected {EXPECTED_AREAS}, "
        f"got {returned_areas}"
    )

    # --------------------------------------------------------
    # Generate synthetic training data
    # --------------------------------------------------------

    training_rows = (
        generate_synthetic_training_data(
            candidate_rows
        )
    )

    print(
        f"Synthetic training rows: "
        f"{len(training_rows)}"
    )

    assert len(training_rows) >= 500, (
        "Training dataset is unexpectedly small."
    )

    # --------------------------------------------------------
    # Validate training data
    # --------------------------------------------------------

    required_training_columns = {
        "source_type",
        "base_area",
        "population_density_per_sqkm",
        "purchasing_power_proxy",
        "household_density_per_sqkm",
        "transit_density_per_sqkm",
        "demand_index",
    }

    actual_training_columns = set(
        training_rows[0].keys()
    )

    assert required_training_columns.issubset(
        actual_training_columns
    ), (
        "Training data is missing required columns."
    )

    for row in training_rows:

        assert row["source_type"] == (
            "synthetic_demo"
        )

        assert (
            10
            <= row["demand_index"]
            <= 100
        )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model = train_demand_model(
        training_rows
    )

    assert model is not None

    # --------------------------------------------------------
    # Run complete pipeline
    # --------------------------------------------------------

    result = run_demand_prediction()

    predictions = result[
        "predictions"
    ]

    assert len(predictions) == 3

    # --------------------------------------------------------
    # Validate predictions
    # --------------------------------------------------------

    areas = set()

    scores = []

    for row in predictions:

        areas.add(
            row["area"].strip().upper()
        )

        assert row[
            "predicted_demand_index"
        ] > 0

        assert (
            10
            <= row["predicted_demand_score"]
            <= 100
        )

        assert row["model_type"] == (
            "HistGradientBoostingRegressor"
        )

        assert row["training_data"] == (
            "synthetic_demo"
        )

        scores.append(
            row["predicted_demand_score"]
        )

        print(
            f"{row['area']}: "
            f"demand_index="
            f"{row['predicted_demand_index']}, "
            f"demand_score="
            f"{row['predicted_demand_score']}"
        )

    assert areas == EXPECTED_AREAS

    assert len(set(scores)) >= 2, (
        "Model produced identical demand "
        "scores for every candidate."
    )

    # --------------------------------------------------------
    # Check output files
    # --------------------------------------------------------

    assert TRAINING_FILE.exists(), (
        "Synthetic training file was not created."
    )

    assert OUTPUT_FILE.exists(), (
        "Demand prediction output was not created."
    )

    print(
        f"\nTraining data:\n"
        f"{TRAINING_FILE}"
    )

    print(
        f"\nPrediction output:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nDEMAND PREDICTION ENGINE PASSED"
    )


if __name__ == "__main__":
    main()
