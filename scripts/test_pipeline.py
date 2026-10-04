from pathlib import Path

from src.pipeline import run_full_pipeline


EXPECTED_AREAS = {
    "BKC",
    "DADAR",
    "VIKHROLI",
}


EXPECTED_OUTPUTS = [
    "candidate_features.csv",
    "market_potential.csv",
    "demand_prediction.csv",
    "demand_capture.csv",
    "store_economics.csv",
    "network_impact.csv",
    "recommendation.csv",
]


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "processed"
)


def main():

    print(
        "Testing complete V2 pipeline...\n"
    )

    result = run_full_pipeline()

    # --------------------------------------------------------
    # Check feature output
    # --------------------------------------------------------

    feature_rows = result[
        "feature_rows"
    ]

    assert len(feature_rows) == 3, (
        f"Expected 3 feature rows, "
        f"got {len(feature_rows)}"
    )

    feature_areas = {
        row["area"].strip().upper()
        for row in feature_rows
    }

    assert feature_areas == EXPECTED_AREAS, (
        f"Expected {EXPECTED_AREAS}, "
        f"got {feature_areas}"
    )

    # --------------------------------------------------------
    # Check Market Potential
    # --------------------------------------------------------

    market_rows = result[
        "market"
    ]["results"]

    assert len(market_rows) == 3

    assert {
        row["area"].strip().upper()
        for row in market_rows
    } == EXPECTED_AREAS

    # --------------------------------------------------------
    # Check Demand Prediction
    # --------------------------------------------------------

    demand_rows = result[
        "demand"
    ]["predictions"]

    assert len(demand_rows) == 3

    assert {
        row["area"].strip().upper()
        for row in demand_rows
    } == EXPECTED_AREAS

    # --------------------------------------------------------
    # Check Demand Capture
    # --------------------------------------------------------

    capture_rows = result[
        "capture"
    ]["results"]

    assert len(capture_rows) == 3

    assert {
        row["area"].strip().upper()
        for row in capture_rows
    } == EXPECTED_AREAS

    # --------------------------------------------------------
    # Check Economics
    # --------------------------------------------------------

    economics_rows = result[
        "economics"
    ]["results"]

    assert len(economics_rows) == 3

    assert {
        row["area"].strip().upper()
        for row in economics_rows
    } == EXPECTED_AREAS

    # --------------------------------------------------------
    # Check Network Impact
    # --------------------------------------------------------

    network_rows = result[
        "network"
    ]["results"]

    assert len(network_rows) == 3

    assert {
        row["area"].strip().upper()
        for row in network_rows
    } == EXPECTED_AREAS

    # --------------------------------------------------------
    # Check Recommendation
    # --------------------------------------------------------

    recommendation = result[
        "recommendation"
    ]

    ranked_rows = recommendation[
        "ranked_candidates"
    ]

    assert len(ranked_rows) == 3

    assert (
        ranked_rows[0]["rank"]
        == 1
    )

    assert (
        ranked_rows[1]["rank"]
        == 2
    )

    assert (
        ranked_rows[2]["rank"]
        == 3
    )

    recommendation_areas = {
        row["area"].strip().upper()
        for row in ranked_rows
    }

    assert (
        recommendation_areas
        == EXPECTED_AREAS
    )

    model_recommendation = (
        recommendation[
            "model_recommendation"
        ]
    )

    assert (
        model_recommendation[
            "area"
        ].strip().upper()
        in EXPECTED_AREAS
    )

    assert (
        model_recommendation[
            "decision_score"
        ]
        is not None
    )

    # --------------------------------------------------------
    # Check physical output files
    # --------------------------------------------------------

    print(
        "\nChecking output files:"
    )

    for filename in EXPECTED_OUTPUTS:

        path = (
            PROCESSED_DIR
            / filename
        )

        assert path.exists(), (
            f"Missing output file: "
            f"{path}"
        )

        assert path.stat().st_size > 0, (
            f"Output file is empty: "
            f"{path}"
        )

        print(
            f"  PASS: {filename}"
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print(
        "\n"
        "MODEL RECOMMENDATION: "
        f"{model_recommendation['area']}"
    )

    print(
        "FINAL DECISION SCORE: "
        f"{model_recommendation['decision_score']}"
    )

    print(
        "\n"
        "FULL V2 PIPELINE PASSED"
    )


if __name__ == "__main__":
    main()
