from src.recommendation import (
    DECISION_WEIGHTS,
    OUTPUT_FILE,
    calculate_decision_scores,
    compare_selected_candidate,
    get_model_recommendation,
    load_all_engine_outputs,
    run_recommendation,
)


EXPECTED_AREAS = {
    "BKC",
    "DADAR",
    "VIKHROLI",
}


def main():

    print("Testing Recommendation Engine...\n")

    # --------------------------------------------------------
    # Validate decision weights
    # --------------------------------------------------------

    assert abs(
        sum(
            DECISION_WEIGHTS.values()
        ) - 1.0
    ) < 0.0001, (
        "Decision weights must sum to 1.0."
    )

    print(
        "Decision weights:"
    )

    for key, value in (
        DECISION_WEIGHTS.items()
    ):

        print(
            f"  {key}: "
            f"{value:.0%}"
        )

    # --------------------------------------------------------
    # Load engine outputs
    # --------------------------------------------------------

    rows = load_all_engine_outputs()

    assert len(rows) == 3, (
        f"Expected 3 candidates, "
        f"got {len(rows)}"
    )

    areas = {
        row["area"].strip().upper()
        for row in rows
    }

    assert areas == EXPECTED_AREAS, (
        f"Expected {EXPECTED_AREAS}, "
        f"got {areas}"
    )

    # --------------------------------------------------------
    # Calculate decision scores
    # --------------------------------------------------------

    scored_rows = (
        calculate_decision_scores(
            rows
        )
    )

    assert len(scored_rows) == 3

    for row in scored_rows:

        assert (
            row["decision_score"]
            is not None
        )

        assert (
            10
            <= row["decision_score"]
            <= 100
        )

        print(
            f"\n{row['area']}:"
            f"\n  market = "
            f"{row['market_potential_score']}"
            f"\n  demand = "
            f"{row['predicted_demand_score']}"
            f"\n  capture = "
            f"{row['captured_demand_score']}"
            f"\n  economics = "
            f"{row['economic_attractiveness_score']}"
            f"\n  network = "
            f"{row['network_impact_score']}"
            f"\n  final decision = "
            f"{row['decision_score']}"
        )

    # --------------------------------------------------------
    # Validate ranking
    # --------------------------------------------------------

    assert (
        [row["rank"] for row in scored_rows]
        == [1, 2, 3]
    )

    scores = [
        row["decision_score"]
        for row in scored_rows
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    ), (
        "Candidates are not sorted "
        "by descending decision score."
    )

    model_recommendation = (
        get_model_recommendation(
            scored_rows
        )
    )

    assert model_recommendation[
        "rank"
    ] == 1

    print(
        "\nMODEL RECOMMENDATION:"
        f" {model_recommendation['area']}"
        f" — {model_recommendation['decision_score']}"
    )

    # --------------------------------------------------------
    # Test selected-candidate comparison
    #
    # We deliberately select DADAR for the test instead of
    # assuming it should be the winner.
    # --------------------------------------------------------

    comparison = (
        compare_selected_candidate(
            scored_rows,
            "DADAR",
        )
    )

    assert (
        comparison[
            "selected_candidate"
        ].upper()
        == "DADAR"
    )

    assert (
        comparison[
            "selected_candidate_rank"
        ]
        in {1, 2, 3}
    )

    assert (
        comparison[
            "selected_candidate_score"
        ]
        is not None
    )

    assert (
        comparison[
            "model_recommendation_score"
        ]
        is not None
    )

    assert comparison[
        "comparison_status"
    ] in {
        "model_agrees_with_selected",
        "model_prefers_alternative",
        "selected_candidate_outperforms_model",
    }

    print(
        "\nSelected candidate: "
        f"{comparison['selected_candidate']}"
    )

    print(
        "Selected candidate rank: "
        f"{comparison['selected_candidate_rank']}"
    )

    print(
        "Selected candidate score: "
        f"{comparison['selected_candidate_score']}"
    )

    print(
        "Model recommendation: "
        f"{comparison['model_recommendation']}"
    )

    print(
        "Score gap: "
        f"{comparison['score_gap']}"
    )

    print(
        "Comparison status: "
        f"{comparison['comparison_status']}"
    )

    # --------------------------------------------------------
    # End-to-end pipeline
    # --------------------------------------------------------

    result = run_recommendation(
        selected_candidate="DADAR"
    )

    assert result[
        "output_path"
    ].exists(), (
        "Recommendation output file "
        "was not created."
    )

    assert (
        len(
            result[
                "ranked_candidates"
            ]
        )
        == 3
    )

    print(
        f"\nRecommendation output:\n"
        f"{result['output_path']}"
    )

    print(
        "\nRECOMMENDATION ENGINE PASSED"
    )


if __name__ == "__main__":
    main()
