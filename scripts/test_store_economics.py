from src.store_economics import (
    CAPTURE_FILE,
    FEATURE_FILE,
    OUTPUT_FILE,
    calculate_store_economics,
    load_csv,
    run_store_economics,
)


EXPECTED_AREAS = {
    "BKC",
    "DADAR",
    "VIKHROLI",
}


def main():

    print("Testing Store Economics Engine...\n")

    # --------------------------------------------------------
    # Load inputs
    # --------------------------------------------------------

    feature_rows = load_csv(
        FEATURE_FILE
    )

    capture_rows = load_csv(
        CAPTURE_FILE
    )

    assert len(feature_rows) == 3, (
        f"Expected 3 feature rows, "
        f"got {len(feature_rows)}"
    )

    assert len(capture_rows) == 3, (
        f"Expected 3 capture rows, "
        f"got {len(capture_rows)}"
    )

    # --------------------------------------------------------
    # Validate candidates
    # --------------------------------------------------------

    feature_areas = {
        row["area"].strip().upper()
        for row in feature_rows
    }

    capture_areas = {
        row["area"].strip().upper()
        for row in capture_rows
    }

    assert feature_areas == EXPECTED_AREAS
    assert capture_areas == EXPECTED_AREAS

    # --------------------------------------------------------
    # Run engine
    # --------------------------------------------------------

    results = calculate_store_economics(
        feature_rows,
        capture_rows,
    )

    assert len(results) == 3

    # --------------------------------------------------------
    # Validate results
    # --------------------------------------------------------

    returned_areas = set()

    for row in results:

        returned_areas.add(
            row["area"].strip().upper()
        )

        assert (
            row["captured_demand_score"]
            is not None
        )

        assert (
            10
            <= row["captured_demand_score"]
            <= 100
        )

        assert (
            10
            <= row["economic_attractiveness_score"]
            <= 100
        )

        assert row[
            "economics_data_quality"
        ] in {
            "complete",
            "partial",
        }

        print(
            f"{row['area']}: "
            f"captured_demand="
            f"{row['captured_demand_score']}, "
            f"annual_rent="
            f"{row['annual_rent']}, "
            f"rent_score="
            f"{row['rent_attractiveness_score']}, "
            f"economics_score="
            f"{row['economic_attractiveness_score']}, "
            f"quality="
            f"{row['economics_data_quality']}"
        )

    assert returned_areas == EXPECTED_AREAS

    # --------------------------------------------------------
    # Verify missing rent does NOT break the engine.
    # --------------------------------------------------------

    for row in results:

        if row["annual_rent"] is None:

            assert row[
                "economics_data_quality"
            ] == "partial"

            assert row[
                "rent_weight_used"
            ] == 0.0

    # --------------------------------------------------------
    # Verify output pipeline
    # --------------------------------------------------------

    result = run_store_economics()

    assert result[
        "output_path"
    ].exists(), (
        "Store Economics output was not created."
    )

    print(
        f"\nOutput written to:\n"
        f"{result['output_path']}"
    )

    print(
        "\nSTORE ECONOMICS ENGINE PASSED"
    )


if __name__ == "__main__":
    main()
