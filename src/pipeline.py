from datetime import datetime, timezone
from pathlib import Path

from src.acquisition import (
    collect_public_data,
    write_raw_payload,
)

from src.features import (
    build_candidate_features,
    write_feature_csv,
)

from src.market_potential import (
    run_market_potential,
)

from src.demand_prediction import (
    run_demand_prediction,
)

from src.demand_capture import (
    run_demand_capture,
)

from src.store_economics import (
    run_store_economics,
)

from src.network_impact import (
    run_network_impact,
)

from src.recommendation import (
    run_recommendation,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "processed"
)


# ============================================================
# HELPERS
# ============================================================

def _now_iso():
    return datetime.now(
        timezone.utc
    ).isoformat()


def _print_stage(
    number,
    total,
    name,
):
    print(
        f"\n{'=' * 60}"
    )

    print(
        f"[{number}/{total}] {name}"
    )

    print(
        f"{'=' * 60}"
    )


# ============================================================
# FULL V2 PIPELINE
# ============================================================

def run_full_pipeline(
    selected_candidate=None,
):
    """
    Run the complete V2 Store Intelligence pipeline.

    Flow:

        Public Data
            ↓
        Raw Data
            ↓
        Feature Engineering
            ↓
        Market Potential
            ↓
        Demand Prediction
            ↓
        Demand Capture
            ↓
        Store Economics
            ↓
        Network Impact
            ↓
        Recommendation
    """

    pipeline_start = _now_iso()

    print(
        "\n"
        "╔══════════════════════════════════════════════════════╗\n"
        "║              STORE INTELLIGENCE V2                  ║\n"
        "║            END-TO-END PIPELINE                     ║\n"
        "╚══════════════════════════════════════════════════════╝"
    )

    # ========================================================
    # 1. PUBLIC DATA ACQUISITION
    # ========================================================

    _print_stage(
        1,
        8,
        "PUBLIC DATA ACQUISITION",
    )

    public_data = collect_public_data()

    print(
        "\nAcquisition summary:"
    )

    for source_name, rows in (
        public_data.items()
    ):

        print(
            f"  {source_name}: "
            f"{len(rows)} rows"
        )

    raw_path = write_raw_payload(
        public_data
    )

    print(
        f"\nRaw data written to:\n"
        f"{raw_path}"
    )

    # ========================================================
    # 2. FEATURE ENGINEERING
    # ========================================================

    _print_stage(
        2,
        8,
        "FEATURE ENGINEERING",
    )

    feature_rows = (
        build_candidate_features(
            public_data
        )
    )

    feature_path = write_feature_csv(
        feature_rows
    )

    print(
        f"\nCandidate feature rows: "
        f"{len(feature_rows)}"
    )

    print(
        f"Feature table:\n"
        f"{feature_path}"
    )

    # ========================================================
    # 3. MARKET POTENTIAL
    # ========================================================

    _print_stage(
        3,
        8,
        "MARKET POTENTIAL",
    )

    market_result = (
        run_market_potential()
    )

    print(
        "\nMarket Potential:"
    )

    for row in market_result[
        "results"
    ]:

        print(
            f"  {row['area']}: "
            f"{row['market_potential_score']}"
        )

    # ========================================================
    # 4. DEMAND PREDICTION
    # ========================================================

    _print_stage(
        4,
        8,
        "DEMAND PREDICTION / ML",
    )

    demand_result = (
        run_demand_prediction()
    )

    print(
        "\nPredicted Demand:"
    )

    for row in demand_result[
        "predictions"
    ]:

        print(
            f"  {row['area']}: "
            f"{row['predicted_demand_score']}"
        )

    # ========================================================
    # 5. DEMAND CAPTURE
    # ========================================================

    _print_stage(
        5,
        8,
        "DEMAND CAPTURE",
    )

    capture_result = (
        run_demand_capture()
    )

    print(
        "\nCaptured Demand:"
    )

    for row in capture_result[
        "results"
    ]:

        print(
            f"  {row['area']}: "
            f"{row['captured_demand_score']}"
        )

    # ========================================================
    # 6. STORE ECONOMICS
    # ========================================================

    _print_stage(
        6,
        8,
        "STORE ECONOMICS",
    )

    economics_result = (
        run_store_economics()
    )

    print(
        "\nStore Economics:"
    )

    for row in economics_result[
        "results"
    ]:

        print(
            f"  {row['area']}: "
            f"score="
            f"{row['economic_attractiveness_score']} "
            f"| quality="
            f"{row['economics_data_quality']}"
        )

    # ========================================================
    # 7. NETWORK IMPACT
    # ========================================================

    _print_stage(
        7,
        8,
        "NETWORK IMPACT / CANNIBALISATION",
    )

    network_result = (
        run_network_impact()
    )

    print(
        "\nNetwork Impact:"
    )

    for row in network_result[
        "results"
    ]:

        print(
            f"  {row['area']}: "
            f"{row['network_impact_score']}"
            f" | cannibalisation pressure="
            f"{row['cannibalisation_pressure']}"
        )

    # ========================================================
    # 8. RECOMMENDATION
    # ========================================================

    _print_stage(
        8,
        8,
        "FINAL RECOMMENDATION",
    )

    recommendation_result = (
        run_recommendation(
            selected_candidate=selected_candidate
        )
    )

    print(
        "\nFINAL RANKING:"
    )

    for row in recommendation_result[
        "ranked_candidates"
    ]:

        print(
            f"  {row['rank']}. "
            f"{row['area']} "
            f"→ "
            f"{row['decision_score']}"
        )

    model_recommendation = (
        recommendation_result[
            "model_recommendation"
        ]
    )

    print(
        "\n"
        "╔══════════════════════════════════════════════════════╗"
    )

    print(
        "║                 MODEL RECOMMENDATION                ║"
    )

    print(
        "╠══════════════════════════════════════════════════════╣"
    )

    print(
        f"║ Location: "
        f"{model_recommendation['area']}"
    )

    print(
        f"║ Score:    "
        f"{model_recommendation['decision_score']}"
    )

    print(
        "╚══════════════════════════════════════════════════════╝"
    )

    # ========================================================
    # USER COMPARISON
    # ========================================================

    comparison = (
        recommendation_result[
            "comparison"
        ]
    )

    if (
        comparison[
            "selected_candidate"
        ]
        is not None
    ):

        print(
            "\nUSER CANDIDATE COMPARISON:"
        )

        print(
            f"  Selected: "
            f"{comparison['selected_candidate']}"
        )

        print(
            f"  Selected Score: "
            f"{comparison['selected_candidate_score']}"
        )

        print(
            f"  Model Recommendation: "
            f"{comparison['model_recommendation']}"
        )

        print(
            f"  Model Score: "
            f"{comparison['model_recommendation_score']}"
        )

        print(
            f"  Status: "
            f"{comparison['comparison_status']}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    pipeline_end = _now_iso()

    print(
        "\n"
        "Pipeline completed successfully."
    )

    print(
        f"Started:  {pipeline_start}"
    )

    print(
        f"Finished: {pipeline_end}"
    )

    print(
        "\nProcessed output directory:"
    )

    print(
        f"{PROCESSED_DIR}"
    )

    return {
        "public_data":
            public_data,

        "feature_rows":
            feature_rows,

        "market":
            market_result,

        "demand":
            demand_result,

        "capture":
            capture_result,

        "economics":
            economics_result,

        "network":
            network_result,

        "recommendation":
            recommendation_result,

        "pipeline_started_at":
            pipeline_start,

        "pipeline_finished_at":
            pipeline_end,
    }


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_full_pipeline()
