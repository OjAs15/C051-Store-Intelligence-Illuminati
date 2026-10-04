from src.network_impact import (
    CAPTURE_FILE,
    FEATURE_FILE,
    OUTPUT_FILE,
    _normalize_existing_stores,
    calculate_cannibalisation,
    calculate_network_impact,
    load_csv,
    run_network_impact,
)


EXPECTED_AREAS = {
    "BKC",
    "DADAR",
    "VIKHROLI",
}


def main():

    print("Testing Network Impact Engine...\n")

    # --------------------------------------------------------
    # Load candidate inputs
    # --------------------------------------------------------

    candidate_rows = load_csv(
        FEATURE_FILE
    )

    capture_rows = load_csv(
        CAPTURE_FILE
    )

    assert len(candidate_rows) == 3, (
        f"Expected 3 candidates, "
        f"got {len(candidate_rows)}"
    )

    assert len(capture_rows) == 3, (
        f"Expected 3 capture rows, "
        f"got {len(capture_rows)}"
    )

    candidate_areas = {
        row["area"].strip().upper()
        for row in candidate_rows
    }

    capture_areas = {
        row["area"].strip().upper()
        for row in capture_rows
    }

    assert candidate_areas == EXPECTED_AREAS
    assert capture_areas == EXPECTED_AREAS

    # --------------------------------------------------------
    # Load fixed existing stores
    # --------------------------------------------------------

    existing_stores = (
        _normalize_existing_stores()
    )

    assert len(existing_stores) >= 1, (
        "No fixed existing stores were loaded."
    )

    print(
        f"Fixed existing stores: "
        f"{len(existing_stores)}"
    )

    for store in existing_stores:

        print(
            f"  {store['name']}: "
            f"lat={store['lat']}, "
            f"lon={store['lon']}, "
            f"weight="
            f"{store['performance_weight']}"
        )

    # --------------------------------------------------------
    # Test pair-level / aggregate cannibalisation
    # --------------------------------------------------------

    cannibalisation = (
        calculate_cannibalisation(
            candidate_rows,
            existing_stores,
        )
    )

    assert len(
        cannibalisation
    ) == 3

    for row in cannibalisation:

        assert (
            row["existing_store_count"]
            == len(existing_stores)
        )

        assert (
            row["cannibalisation_pressure"]
            >= 0
        )

        assert (
            row["strongest_overlap_score"]
            is not None
        )

        assert (
            0
            <= row["strongest_overlap_score"]
            <= 1
        )

        print(
            f"\n{row['area']}:"
            f"\n  nearest existing = "
            f"{row['nearest_existing_store']}"
            f"\n  nearest distance = "
            f"{row['nearest_existing_store_distance_km']} km"
            f"\n  strongest overlap = "
            f"{row['strongest_overlap_store']}"
            f"\n  overlap score = "
            f"{row['strongest_overlap_score']}"
            f"\n  cannibalisation pressure = "
            f"{row['cannibalisation_pressure']}"
        )

    # --------------------------------------------------------
    # Test network impact
    # --------------------------------------------------------

    results = calculate_network_impact(
        candidate_rows,
        capture_rows,
        existing_stores,
    )

    assert len(results) == 3

    returned_areas = set()

    network_scores = []

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
            row["cannibalisation_score"]
            is not None
        )

        assert (
            10
            <= row["cannibalisation_score"]
            <= 100
        )

        assert (
            row[
                "cannibalisation_protection_score"
            ]
            is not None
        )

        assert (
            10
            <= row[
                "cannibalisation_protection_score"
            ]
            <= 100
        )

        assert (
            row["network_impact_score"]
            is not None
        )

        assert (
            10
            <= row["network_impact_score"]
            <= 100
        )

        assert (
            row[
                "incremental_network_value_proxy"
            ]
            is not None
        )

        network_scores.append(
            row["network_impact_score"]
        )

        print(
            f"\n{row['area']}:"
            f"\n  captured demand = "
            f"{row['captured_demand_score']}"
            f"\n  cannibalisation score = "
            f"{row['cannibalisation_score']}"
            f"\n  protection score = "
            f"{row['cannibalisation_protection_score']}"
            f"\n  network impact = "
            f"{row['network_impact_score']}"
            f"\n  network value proxy = "
            f"{row['incremental_network_value_proxy']}"
        )

    assert returned_areas == EXPECTED_AREAS

    # The engine should normally produce some differentiation.
    assert len(
        set(network_scores)
    ) >= 2, (
        "Network Impact produced identical "
        "scores for all candidates."
    )

    # --------------------------------------------------------
    # End-to-end output test
    # --------------------------------------------------------

    result = run_network_impact()

    assert result[
        "output_path"
    ].exists(), (
        "Network Impact output file was not created."
    )

    print(
        f"\nOutput written to:\n"
        f"{result['output_path']}"
    )

    print(
        "\nNETWORK IMPACT ENGINE PASSED"
    )


if __name__ == "__main__":
    main()
