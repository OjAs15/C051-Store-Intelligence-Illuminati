from src.market_potential import (
    FEATURE_FILE,
    calculate_market_potential,
    load_feature_table,
    run_market_potential,
)


EXPECTED_AREAS = {
    "BKC",
    "DADAR",
    "VIKHROLI",
}


REQUIRED_COLUMNS = {
    "area_id",
    "area",
    "population_density_per_sqkm",
    "purchasing_power_proxy",
    "household_density_per_sqkm",
    "population_market_index",
    "purchasing_power_market_index",
    "household_market_index",
    "market_potential_score",
    "market_data_quality",
}


def main():

    print("Testing Market Potential Engine...\n")

    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------

    rows = load_feature_table(
        FEATURE_FILE
    )

    assert len(rows) == 3, (
        f"Expected 3 candidate rows, "
        f"got {len(rows)}"
    )

    returned_areas = {
        row["area"].strip().upper()
        for row in rows
    }

    assert returned_areas == EXPECTED_AREAS, (
        f"Expected {EXPECTED_AREAS}, "
        f"got {returned_areas}"
    )

    # --------------------------------------------------------
    # Run engine
    # --------------------------------------------------------

    results = calculate_market_potential(
        rows
    )

    assert len(results) == 3, (
        f"Expected 3 results, "
        f"got {len(results)}"
    )

    # --------------------------------------------------------
    # Validate output structure
    # --------------------------------------------------------

    actual_columns = set(
        results[0].keys()
    )

    missing_columns = (
        REQUIRED_COLUMNS
        - actual_columns
    )

    assert not missing_columns, (
        f"Missing output columns: "
        f"{missing_columns}"
    )

    # --------------------------------------------------------
    # Validate scores
    # --------------------------------------------------------

    scores = []

    for row in results:

        score = row[
            "market_potential_score"
        ]

        assert score is not None, (
            f"Missing Market Potential score "
            f"for {row['area']}"
        )

        assert 10 <= score <= 100, (
            f"Invalid score for "
            f"{row['area']}: {score}"
        )

        scores.append(score)

        print(
            f"{row['area']}: "
            f"population_index="
            f"{row['population_market_index']:.2f}, "
            f"purchasing_power_index="
            f"{row['purchasing_power_market_index']:.2f}, "
            f"household_index="
            f"{row['household_market_index']:.2f}, "
            f"market_potential="
            f"{row['market_potential_score']}"
        )

    # --------------------------------------------------------
    # Make sure candidates are actually differentiated
    # --------------------------------------------------------

    assert len(set(scores)) >= 2, (
        "Market Potential Engine produced identical "
        "scores for all candidates."
    )

    # --------------------------------------------------------
    # Test complete write pipeline
    # --------------------------------------------------------

    result = run_market_potential()

    assert result["output_path"].exists(), (
        "Market Potential output file was not created."
    )

    print(
        f"\nOutput written to:\n"
        f"{result['output_path']}"
    )

    print("\nMARKET POTENTIAL ENGINE PASSED")


if __name__ == "__main__":
    main()
