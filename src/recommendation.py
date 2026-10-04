import csv
from pathlib import Path


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

MARKET_FILE = (
    PROCESSED_DIR
    / "market_potential.csv"
)

DEMAND_FILE = (
    PROCESSED_DIR
    / "demand_prediction.csv"
)

CAPTURE_FILE = (
    PROCESSED_DIR
    / "demand_capture.csv"
)

ECONOMICS_FILE = (
    PROCESSED_DIR
    / "store_economics.csv"
)

NETWORK_FILE = (
    PROCESSED_DIR
    / "network_impact.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "recommendation.csv"
)


# ============================================================
# DECISION WEIGHTS
# ============================================================

# These weights define the POC's final decision framework.
#
# Market Potential:
#   underlying opportunity
#
# Demand Prediction:
#   expected store demand
#
# Demand Capture:
#   realistic demand after accessibility + competition
#
# Store Economics:
#   demand vs rent/cost attractiveness
#
# Network Impact:
#   effect on the existing store network
#
# Demand Capture and Network Impact intentionally receive
# higher weights because the core business question is not
# simply "which market is biggest?" but "which site creates
# the most attractive incremental network opportunity?"

DECISION_WEIGHTS = {
    "market_potential": 0.15,
    "demand_prediction": 0.15,
    "demand_capture": 0.30,
    "store_economics": 0.15,
    "network_impact": 0.25,
}


# ============================================================
# HELPERS
# ============================================================

def _to_float(value):
    """
    Safely convert a value to float.
    """

    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_csv(path):
    """
    Load a CSV file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required recommendation input "
            f"not found: {path}"
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
            f"Input CSV is empty: {path}"
        )

    return rows


def _index_by_area_id(rows):
    """
    Convert a candidate table into an area_id lookup.
    """

    output = {}

    for row in rows:

        area_id = row.get(
            "area_id"
        )

        if not area_id:
            continue

        output[area_id] = row

    return output


def _weighted_average(
    components,
    weights,
):
    """
    Calculate weighted score using only valid components.

    Missing components are excluded and the remaining weights
    are renormalized.

    This is important because missing rent should not cause a
    candidate to receive an artificial zero.
    """

    usable = []

    for key, weight in weights.items():

        value = components.get(key)

        if value is None:
            continue

        usable.append(
            (
                value,
                weight,
            )
        )

    if not usable:
        return None

    total_weight = sum(
        weight
        for _, weight in usable
    )

    if total_weight <= 0:
        return None

    return sum(
        value
        * (
            weight
            / total_weight
        )
        for value, weight in usable
    )


def _format_score(value):
    """
    Format score safely for generated explanation text.
    """

    if value is None:
        return "N/A"

    return f"{value:.1f}"


# ============================================================
# DATA ASSEMBLY
# ============================================================

def load_all_engine_outputs():
    """
    Load all five engine outputs and combine them by area_id.
    """

    market_rows = load_csv(
        MARKET_FILE
    )

    demand_rows = load_csv(
        DEMAND_FILE
    )

    capture_rows = load_csv(
        CAPTURE_FILE
    )

    economics_rows = load_csv(
        ECONOMICS_FILE
    )

    network_rows = load_csv(
        NETWORK_FILE
    )

    market = _index_by_area_id(
        market_rows
    )

    demand = _index_by_area_id(
        demand_rows
    )

    capture = _index_by_area_id(
        capture_rows
    )

    economics = _index_by_area_id(
        economics_rows
    )

    network = _index_by_area_id(
        network_rows
    )

    area_ids = (
        set(market.keys())
        | set(demand.keys())
        | set(capture.keys())
        | set(economics.keys())
        | set(network.keys())
    )

    if not area_ids:
        raise ValueError(
            "No candidate areas were found "
            "across engine outputs."
        )

    combined = []

    for area_id in sorted(
        area_ids
    ):

        market_row = market.get(
            area_id,
            {}
        )

        demand_row = demand.get(
            area_id,
            {}
        )

        capture_row = capture.get(
            area_id,
            {}
        )

        economics_row = economics.get(
            area_id,
            {}
        )

        network_row = network.get(
            area_id,
            {}
        )

        area = (
            market_row.get("area")
            or demand_row.get("area")
            or capture_row.get("area")
            or economics_row.get("area")
            or network_row.get("area")
            or area_id
        )

        combined.append(
            {
                "area_id": area_id,
                "area": area,

                "market_potential_score":
                    _to_float(
                        market_row.get(
                            "market_potential_score"
                        )
                    ),

                "predicted_demand_score":
                    _to_float(
                        demand_row.get(
                            "predicted_demand_score"
                        )
                    ),

                "captured_demand_score":
                    _to_float(
                        capture_row.get(
                            "captured_demand_score"
                        )
                    ),

                "economic_attractiveness_score":
                    _to_float(
                        economics_row.get(
                            "economic_attractiveness_score"
                        )
                    ),

                "economics_data_quality":
                    economics_row.get(
                        "economics_data_quality",
                        "unknown",
                    ),

                "rent_attractiveness_score":
                    _to_float(
                        economics_row.get(
                            "rent_attractiveness_score"
                        )
                    ),

                "network_impact_score":
                    _to_float(
                        network_row.get(
                            "network_impact_score"
                        )
                    ),

                "incremental_network_value_proxy":
                    _to_float(
                        network_row.get(
                            "incremental_network_value_proxy"
                        )
                    ),

                "cannibalisation_pressure":
                    _to_float(
                        network_row.get(
                            "cannibalisation_pressure"
                        )
                    ),

                "nearest_existing_store":
                    network_row.get(
                        "nearest_existing_store"
                    ),

                "nearest_existing_store_distance_km":
                    _to_float(
                        network_row.get(
                            "nearest_existing_store_distance_km"
                        )
                    ),
            }
        )

    return combined


# ============================================================
# FINAL DECISION SCORE
# ============================================================

def calculate_decision_scores(
    rows
):
    """
    Calculate the final multi-engine decision score.

    Base formula:

      15% Market Potential
      15% Demand Prediction
      30% Demand Capture
      15% Store Economics
      25% Network Impact

    When a component is unavailable, its weight is removed
    and the remaining weights are renormalized.

    When Store Economics is marked "partial", the economics
    component is excluded from the final decision score because
    the current rent input is incomplete.
    """

    results = []

    for row in rows:

        components = {
            "market_potential":
                row[
                    "market_potential_score"
                ],

            "demand_prediction":
                row[
                    "predicted_demand_score"
                ],

            "demand_capture":
                row[
                    "captured_demand_score"
                ],

            "store_economics":
                row[
                    "economic_attractiveness_score"
                ],

            "network_impact":
                row[
                    "network_impact_score"
                ],
        }

        weights = (
            DECISION_WEIGHTS.copy()
        )

        # ----------------------------------------------------
        # Do not let partial economics influence the final
        # score as though verified financial data exists.
        # ----------------------------------------------------

        if (
            row[
                "economics_data_quality"
            ]
            != "complete"
        ):
            weights[
                "store_economics"
            ] = 0.0

        decision_score = (
            _weighted_average(
                components,
                weights,
            )
        )

        # ----------------------------------------------------
        # Effective weights after missing/partial inputs.
        # ----------------------------------------------------

        available_weight = sum(
            weight
            for key, weight in weights.items()
            if components.get(key)
            is not None
        )

        effective_weights = {}

        if available_weight > 0:

            for key, weight in weights.items():

                if (
                    components.get(key)
                    is not None
                ):

                    effective_weights[
                        key
                    ] = round(
                        weight
                        / available_weight,
                        4,
                    )

                else:

                    effective_weights[
                        key
                    ] = 0.0

        else:

            effective_weights = {
                key: 0.0
                for key in weights
            }

        results.append(
            {
                **row,

                "decision_score":
                    (
                        round(
                            decision_score,
                            2,
                        )
                        if decision_score
                        is not None
                        else None
                    ),

                "effective_market_weight":
                    effective_weights[
                        "market_potential"
                    ],

                "effective_demand_weight":
                    effective_weights[
                        "demand_prediction"
                    ],

                "effective_capture_weight":
                    effective_weights[
                        "demand_capture"
                    ],

                "effective_economics_weight":
                    effective_weights[
                        "store_economics"
                    ],

                "effective_network_weight":
                    effective_weights[
                        "network_impact"
                    ],
            }
        )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    results.sort(
        key=lambda row: (
            row["decision_score"]
            if row["decision_score"]
            is not None
            else -1
        ),
        reverse=True,
    )

    for rank, row in enumerate(
        results,
        start=1,
    ):

        row["rank"] = rank

    return results


# ============================================================
# EXPLAINABILITY
# ============================================================

def add_explanations(
    rows
):
    """
    Add concise business-facing explanation fields.
    """

    score_fields = {
        "Market Potential":
            "market_potential_score",

        "Demand Prediction":
            "predicted_demand_score",

        "Demand Capture":
            "captured_demand_score",

        "Store Economics":
            "economic_attractiveness_score",

        "Network Impact":
            "network_impact_score",
    }

    for row in rows:

        available_components = []

        for label, field in score_fields.items():

            value = row.get(field)

            if value is None:
                continue

            # Ignore partial economics from the
            # "strength" / "risk" ranking.
            if (
                field
                == "economic_attractiveness_score"
                and row[
                    "economics_data_quality"
                ]
                != "complete"
            ):
                continue

            available_components.append(
                (
                    label,
                    value,
                )
            )

        if available_components:

            strongest = max(
                available_components,
                key=lambda item: item[1],
            )

            weakest = min(
                available_components,
                key=lambda item: item[1],
            )

            row[
                "primary_strength"
            ] = (
                f"{strongest[0]} "
                f"({strongest[1]:.1f})"
            )

            row[
                "primary_risk"
            ] = (
                f"{weakest[0]} "
                f"({weakest[1]:.1f})"
            )

        else:

            row[
                "primary_strength"
            ] = "Insufficient data"

            row[
                "primary_risk"
            ] = "Insufficient data"

        # ----------------------------------------------------
        # Network-specific explanation
        # ----------------------------------------------------

        nearest_store = row.get(
            "nearest_existing_store"
        )

        nearest_distance = row.get(
            "nearest_existing_store_distance_km"
        )

        if (
            nearest_store
            and nearest_distance is not None
        ):

            row[
                "network_explanation"
            ] = (
                f"Nearest fixed store: "
                f"{nearest_store} "
                f"({nearest_distance:.2f} km)"
            )

        else:

            row[
                "network_explanation"
            ] = (
                "Existing-store overlap data unavailable"
            )

        # ----------------------------------------------------
        # Economics explanation
        # ----------------------------------------------------

        if (
            row[
                "economics_data_quality"
            ]
            == "complete"
        ):

            row[
                "economics_explanation"
            ] = (
                "Demand and rent inputs available."
            )

        else:

            row[
                "economics_explanation"
            ] = (
                "Rent input pending; economics "
                "excluded from final score."
            )

    return rows


# ============================================================
# MODEL RECOMMENDATION
# ============================================================

def get_model_recommendation(
    rows
):
    """
    Return the highest-ranked candidate.
    """

    if not rows:
        raise ValueError(
            "No recommendation rows available."
        )

    top = rows[0]

    return {
        "area_id":
            top["area_id"],

        "area":
            top["area"],

        "decision_score":
            top["decision_score"],

        "rank":
            top["rank"],
    }


# ============================================================
# USER CANDIDATE COMPARISON
# ============================================================

def compare_selected_candidate(
    rows,
    selected_candidate,
):
    """
    Compare the model recommendation against a candidate
    chosen by the user.

    selected_candidate may be:
      - area name
      - area_id
    """

    if (
        selected_candidate is None
        or str(selected_candidate).strip() == ""
    ):
        return {
            "selected_candidate":
                None,

            "selected_candidate_rank":
                None,

            "selected_candidate_score":
                None,

            "model_recommendation":
                rows[0]["area"],

            "model_recommendation_score":
                rows[0]["decision_score"],

            "score_gap":
                None,

            "comparison_status":
                "no_candidate_selected",
        }

    selected_text = (
        str(selected_candidate)
        .strip()
        .upper()
    )

    selected_row = None

    for row in rows:

        if (
            row["area"].strip().upper()
            == selected_text
            or
            row["area_id"].strip().upper()
            == selected_text
        ):

            selected_row = row
            break

    if selected_row is None:

        raise ValueError(
            f"Selected candidate "
            f"'{selected_candidate}' was not found."
        )

    model_row = rows[0]

    score_gap = (
        model_row["decision_score"]
        - selected_row["decision_score"]
    )

    if (
        model_row["area_id"]
        == selected_row["area_id"]
    ):

        status = (
            "model_agrees_with_selected"
        )

    elif score_gap > 0:

        status = (
            "model_prefers_alternative"
        )

    else:

        status = (
            "selected_candidate_outperforms_model"
        )

    return {
        "selected_candidate":
            selected_row["area"],

        "selected_candidate_rank":
            selected_row["rank"],

        "selected_candidate_score":
            selected_row["decision_score"],

        "model_recommendation":
            model_row["area"],

        "model_recommendation_score":
            model_row["decision_score"],

        "score_gap":
            round(
                score_gap,
                2,
            ),

        "comparison_status":
            status,
    }


# ============================================================
# CSV WRITER
# ============================================================

def write_recommendation_csv(
    rows,
    output_path=OUTPUT_FILE,
):
    """
    Write ranked recommendation table.
    """

    if not rows:
        raise ValueError(
            "Cannot write empty recommendation output."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
    )

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
# END-TO-END RECOMMENDATION
# ============================================================

def run_recommendation(
    selected_candidate=None,
):
    """
    Complete recommendation pipeline.

    Returns:
      - ranked candidate rows
      - model recommendation
      - selected-candidate comparison
      - output file path
    """

    rows = load_all_engine_outputs()

    scored_rows = (
        calculate_decision_scores(
            rows
        )
    )

    scored_rows = add_explanations(
        scored_rows
    )

    model_recommendation = (
        get_model_recommendation(
            scored_rows
        )
    )

    comparison = (
        compare_selected_candidate(
            scored_rows,
            selected_candidate,
        )
    )

    output_path = (
        write_recommendation_csv(
            scored_rows
        )
    )

    return {
        "ranked_candidates":
            scored_rows,

        "model_recommendation":
            model_recommendation,

        "comparison":
            comparison,

        "output_path":
            output_path,
    }


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    result = run_recommendation()

    print(
        "Recommendation Engine completed.\n"
    )

    print(
        f"Output file:\n"
        f"{result['output_path']}\n"
    )

    print(
        "Ranked candidates:"
    )

    for row in result[
        "ranked_candidates"
    ]:

        print(
            f"{row['rank']}. "
            f"{row['area']} — "
            f"Decision Score = "
            f"{_format_score(row['decision_score'])}"
        )

    print(
        "\nMODEL RECOMMENDATION: "
        f"{result['model_recommendation']['area']}"
    )
