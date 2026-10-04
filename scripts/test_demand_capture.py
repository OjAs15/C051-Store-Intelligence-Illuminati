from src.demand_capture import (
    DEMAND_FILE,
    FEATURE_FILE,
    OUTPUT_FILE,
    calculate_accessibility_score,
    calculate_competition_score,
    calculate_demand_capture,
    load_csv,
    run_demand_capture,
)


EXPECTED_AREAS = {
    "BKC",
    "DADAR",
    "VIKHROLI",
}


def main():

    print("Testing Demand Capture Engine...\n")

    # --------------------------------------------------------
    # Load inputs
    # --------------------------------------------------------

    feature_rows = load_csv(
        FEATURE_FILE
    )

    demand_rows = load_csv(
        DEMAND_FILE
    )

    assert len(feature_rows) == 3, (
        f"Expected 3 feature rows, "
        f"got {len(feature_rows)}"
    )

    assert len(demand_rows) == 3, (
        f"Expected 3 demand rows, "
        f"got {len(demand_rows)}"
    )

    feature_areas = {
        row["area"].strip().upper()
        for row in feature_rows
    }

    demand_areas = {
        row["area"].strip().upper()
        for row in demand_rows
    }

    assert feature_areas == EXPECTED_AREAS
    assert demand_areas == EXPECTED_AREAS

    # --------------------------------------------------------
    # Test accessibility component
    # --------------------------------------------------------

    accessibility = calculate_accessibility_score(
        feature_rows
    )

    assert len(accessibility) == 3

    for row in accessibility:

        assert row["accessibility_score"] is not None

        assert (
            0
            <= row["accessibility_score"]
            <= 100
        )

        assert (
            row["accessibility_factor"]
            >= 0.80
        )

        assert (
            row["accessibility_factor"]
            <= 1.20
        )

    print(
        "Accessibility adjustment:"
    )

    for row in accessibility:

        print(
            f"  {row['area']}: "
            f"score="
            f"{row['accessibility_score']}, "
            f"factor="
            f"{row['accessibility_factor']}"
        )

    # --------------------------------------------------------
    # Test competition component
    # --------------------------------------------------------

    competition = calculate_competition_score(
        feature_rows
    )

    assert len(competition) == 3

    for row in competition:

        assert row["competition_score"] is not None

        assert (
            0
            <= row["competition_score"]
            <= 100
        )

        assert (
            row["competition_factor"]
            >= 0.80
        )

        assert (
            row["competition_factor"]
            <= 1.10
        )

    print(
        "\nCompetition adjustment:"
    )

    for row in competition:

        print(
            f"  {row['area']}: "
            f"score="
            f"{row['competition_score']}, "
            f"factor="
            f"{row['competition_factor']}"
        )

    # --------------------------------------------------------
    # Run complete engine
    # --------------------------------------------------------

    results = calculate_demand_capture(
        feature_rows,
        demand_rows,
    )

    assert len(results) == 3

    # --------------------------------------------------------
    # Validate results
    # --------------------------------------------------------

    scores = []

    returned_areas = set()

    for row in results:

        returned_areas.add(
            row["area"].strip().upper()
        )

        assert (
            row["predicted_demand_index"]
            > 0
        )

        assert (
            row["accessibility_factor"]
            >= 0.80
        )

        assert (
            row["accessibility_factor"]
            <= 1.20
        )

        assert (
            row["competition_factor"]
            >= 0.80
        )

        assert (
            row["competition_factor"]
            <= 1.10
        )

        assert (
            row["captured_demand_index"]
            > 0
        )

        assert (
            10
            <= row["captured_demand_score"]
            <= 100
        )

        scores.append(
            row["captured_demand_score"]
        )

        print(
            f"\n{row['area']}:"
            f"\n  predicted demand = "
            f"{row['predicted_demand_index']}"
            f"\n  accessibility factor = "
            f"{row['accessibility_factor']}"
            f"\n  competition factor = "
            f"{row['competition_factor']}"
            f"\n  captured demand index = "
            f"{row['captured_demand_index']}"
            f"\n  capture score = "
            f"{row['captured_demand_score']}"
        )

    assert returned_areas == EXPECTED_AREAS

    # The candidates should normally be differentiated.
    assert len(set(scores)) >= 2, (
        "Demand Capture produced identical "
        "scores for all candidates."
    )

    # --------------------------------------------------------
    # Test output pipeline
    # --------------------------------------------------------

    result = run_demand_capture()

    assert result["output_path"].exists(), (
        "Demand Capture output file was not created."
    )

    print(
        f"\nOutput written to:\n"
        f"{result['output_path']}"
    )

    print(
        "\nDEMAND CAPTURE ENGINE PASSED"
    )


if __name__ == "__main__":
    main()
