import ast
from pathlib import Path

import pandas as pd
import streamlit as st

from src.demand_capture import load_csv as load_capture_csv
from src.network_impact import (
    _normalize_existing_stores,
    calculate_network_impact,
    load_csv as load_network_csv,
    write_network_impact_csv,
)
from src.pipeline import run_full_pipeline
from src.recommendation import run_recommendation


# ============================================================
# OPTIONAL MAP DEPENDENCY
# ============================================================

try:
    import pydeck as pdk

    PYDECK_AVAILABLE = True

except ImportError:
    PYDECK_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "processed"
)

FEATURE_FILE = (
    PROCESSED_DIR
    / "candidate_features.csv"
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

RECOMMENDATION_FILE = (
    PROCESSED_DIR
    / "recommendation.csv"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Store Intelligence",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f7f8fa;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    .main-title {
        font-size: 34px;
        font-weight: 700;
        letter-spacing: -0.8px;
        margin-bottom: 4px;
    }

    .sub-title {
        font-size: 15px;
        color: #667085;
        margin-bottom: 22px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 8px;
        margin-bottom: 4px;
    }

    .section-subtitle {
        font-size: 14px;
        color: #667085;
        margin-bottom: 18px;
    }

    .insight-box {
        padding: 16px 18px;
        border-radius: 12px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .recommendation-box {
        padding: 22px;
        border-radius: 14px;
        background: #111827;
        color: white;
        margin-top: 12px;
        margin-bottom: 18px;
    }

    .recommendation-location {
        font-size: 30px;
        font-weight: 700;
    }

    .small-muted {
        color: #667085;
        font-size: 13px;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        padding: 14px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "pipeline_loaded" not in st.session_state:
    st.session_state.pipeline_loaded = False

if "last_pipeline_run" not in st.session_state:
    st.session_state.last_pipeline_run = None

if "fixed_store_ids" not in st.session_state:
    st.session_state.fixed_store_ids = None

if "selected_candidate" not in st.session_state:
    st.session_state.selected_candidate = None


# ============================================================
# HELPERS
# ============================================================

def file_exists(path):
    return path.exists() and path.stat().st_size > 0


def load_dataframe(path):
    if not file_exists(path):
        return pd.DataFrame()

    return pd.read_csv(path)


def load_all_data():
    return {
        "features": load_dataframe(FEATURE_FILE),
        "market": load_dataframe(MARKET_FILE),
        "demand": load_dataframe(DEMAND_FILE),
        "capture": load_dataframe(CAPTURE_FILE),
        "economics": load_dataframe(ECONOMICS_FILE),
        "network": load_dataframe(NETWORK_FILE),
        "recommendation": load_dataframe(
            RECOMMENDATION_FILE
        ),
    }


def has_complete_pipeline():
    return all(
        file_exists(path)
        for path in [
            FEATURE_FILE,
            MARKET_FILE,
            DEMAND_FILE,
            CAPTURE_FILE,
            ECONOMICS_FILE,
            NETWORK_FILE,
            RECOMMENDATION_FILE,
        ]
    )


def get_existing_stores():
    """
    Load the fixed-store configuration used by the network
    engine.
    """

    try:
        return _normalize_existing_stores()

    except Exception as exc:

        st.error(
            "Could not load existing-store configuration."
        )

        st.exception(exc)

        return []


def get_candidate_names(data):
    features = data["features"]

    if features.empty:
        return []

    return (
        features["area"]
        .dropna()
        .astype(str)
        .tolist()
    )


def safe_round(value, digits=1):
    if value is None:
        return "—"

    try:
        return round(float(value), digits)

    except (TypeError, ValueError):
        return "—"


def format_factor(value):
    if value is None:
        return "—"

    try:
        return f"{float(value):.2f}x"

    except (TypeError, ValueError):
        return "—"


def score_delta(candidate_score, benchmark_score):
    if (
        candidate_score is None
        or benchmark_score is None
    ):
        return "—"

    delta = (
        float(candidate_score)
        - float(benchmark_score)
    )

    return f"{delta:+.1f}"


def friendly_label(value):
    return (
        str(value)
        .replace("_", " ")
        .title()
    )


def parse_overlap_detail(value):
    """
    Convert the stored list-string into a readable dataframe.
    """

    if value is None:
        return pd.DataFrame()

    try:
        parsed = ast.literal_eval(
            str(value)
        )

        if not isinstance(parsed, list):
            return pd.DataFrame()

        return pd.DataFrame(parsed)

    except Exception:
        return pd.DataFrame()


# ============================================================
# APPLY FIXED STORE SELECTION
# ============================================================

def apply_fixed_stores(selected_ids):
    """
    Recalculate only the network-impact and recommendation
    layers using the fixed stores chosen in the UI.

    This avoids recollecting public data when the user changes
    which existing stores are treated as fixed.
    """

    features = load_network_csv(
        FEATURE_FILE
    )

    capture = load_capture_csv(
        CAPTURE_FILE
    )

    configured_stores = (
        _normalize_existing_stores()
    )

    selected_ids = set(
        selected_ids
    )

    selected_stores = [
        store
        for store in configured_stores
        if store["id"] in selected_ids
    ]

    if not selected_stores:
        raise ValueError(
            "At least one existing store must be fixed."
        )

    network_results = calculate_network_impact(
        features,
        capture,
        selected_stores,
    )

    write_network_impact_csv(
        network_results,
        NETWORK_FILE,
    )

    run_recommendation(
        selected_candidate=st.session_state.selected_candidate
    )

    st.session_state.fixed_store_ids = (
        selected_ids
    )


# ============================================================
# INITIAL DATA LOAD
# ============================================================

data = load_all_data()

if has_complete_pipeline():
    st.session_state.pipeline_loaded = True


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:22px;
            font-weight:700;
            margin-bottom:18px;
        ">
            Store Intelligence
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        [
            "Network & Locations",
            "Market Potential",
            "Demand Prediction",
            "Demand Capture",
            "What-if Simulator",
            "Store Economics",
            "Network Impact",
            "Recommendation",
        ],
    )

    st.divider()

    st.caption(
        "V2 Proof of Concept"
    )

    if st.button(
        "↻ Refresh V2 Data",
        use_container_width=True,
    ):

        with st.spinner(
            "Running public-data pipeline..."
        ):

            try:

                result = run_full_pipeline()

                st.session_state.pipeline_loaded = True

                st.session_state.last_pipeline_run = (
                    result[
                        "pipeline_finished_at"
                    ]
                )

                # Refresh in-memory data.
                data = load_all_data()

                # Reset fixed-store override to canonical
                # configuration after a full refresh.
                st.session_state.fixed_store_ids = None

                st.success(
                    "V2 refresh completed."
                )

                st.rerun()

            except Exception as exc:

                st.error(
                    "Pipeline refresh failed."
                )

                st.exception(exc)

    st.divider()

    st.markdown(
        """
        <div class="small-muted">
        Decision model:<br>
        Market → Demand → Capture → Economics → Network
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# REQUIRE PIPELINE
# ============================================================

if not st.session_state.pipeline_loaded:

    st.markdown(
        '<div class="main-title">Store Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sub-title">
        Location expansion intelligence for the next store.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Run the V2 pipeline from the sidebar to load "
        "the live public-data analysis."
    )

    st.stop()


# Refresh dataframe references after any actions.
data = load_all_data()


# ============================================================
# EXISTING STORES
# ============================================================

existing_stores = get_existing_stores()

if st.session_state.fixed_store_ids is None:

    st.session_state.fixed_store_ids = {
        store["id"]
        for store in existing_stores
    }


# ============================================================
# PAGE 1 — NETWORK & LOCATIONS
# ============================================================

if page == "Network & Locations":

    st.markdown(
        '<div class="section-title">'
        'Network & Locations'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-subtitle">
        Define the fixed existing network and compare the
        three candidate expansion locations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Fixed stores
    # --------------------------------------------------------

    st.markdown(
        "### 1. Fixed Existing Stores"
    )

    store_options = {
        store["name"]: store["id"]
        for store in existing_stores
    }

    default_names = [
        store["name"]
        for store in existing_stores
        if store["id"]
        in st.session_state.fixed_store_ids
    ]

    selected_names = st.multiselect(
        "Which existing stores should remain fixed?",
        options=list(
            store_options.keys()
        ),
        default=default_names,
        max_selections=2,
    )

    col_apply, col_reset = st.columns(
        [1, 1]
    )

    with col_apply:

        if st.button(
            "Apply Fixed Stores",
            use_container_width=True,
        ):

            if not selected_names:

                st.error(
                    "Select at least one existing store."
                )

            else:

                selected_ids = {
                    store_options[name]
                    for name in selected_names
                }

                with st.spinner(
                    "Recalculating network impact..."
                ):

                    try:

                        apply_fixed_stores(
                            selected_ids
                        )

                        data = load_all_data()

                        st.success(
                            "Fixed-store network updated."
                        )

                        st.rerun()

                    except Exception as exc:

                        st.error(
                            "Could not update fixed stores."
                        )

                        st.exception(exc)

    with col_reset:

        if st.button(
            "Reset to Configured Network",
            use_container_width=True,
        ):

            configured_ids = {
                store["id"]
                for store in existing_stores
            }

            with st.spinner(
                "Resetting network..."
            ):

                try:

                    apply_fixed_stores(
                        configured_ids
                    )

                    data = load_all_data()

                    st.success(
                        "Network reset."
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        "Could not reset network."
                    )

                    st.exception(exc)

    st.divider()

    # --------------------------------------------------------
    # KPI cards
    # --------------------------------------------------------

    candidates = data[
        "features"
    ]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Existing Stores Fixed",
        len(
            st.session_state.fixed_store_ids
        ),
    )

    col2.metric(
        "Candidate Locations",
        len(candidates),
    )

    if not candidates.empty:

        radius = candidates[
            "catchment_radius_km"
        ].iloc[0]

    else:

        radius = "—"

    col3.metric(
        "Catchment Radius",
        f"{radius} km"
        if radius != "—"
        else "—",
    )

    col4.metric(
        "Retailer",
        "Croma",
    )

    # --------------------------------------------------------
    # Map
    # --------------------------------------------------------

    st.markdown(
        "### 2. Location Network"
    )

    candidate_points = candidates[
        [
            "area",
            "latitude",
            "longitude",
        ]
    ].copy()

    candidate_points["type"] = (
        "Candidate"
    )

    candidate_points["marker_size"] = (
        180
    )

    fixed_ids = (
        st.session_state.fixed_store_ids
    )

    existing_points = pd.DataFrame(
        [
            {
                "area": store["name"],
                "latitude": store["lat"],
                "longitude": store["lon"],
                "type": "Existing Store",
                "marker_size": 220,
            }
            for store in existing_stores
            if store["id"] in fixed_ids
        ]
    )

    map_data = pd.concat(
        [
            candidate_points,
            existing_points,
        ],
        ignore_index=True,
    )

    if (
        PYDECK_AVAILABLE
        and not map_data.empty
    ):

        center_lat = map_data[
            "latitude"
        ].mean()

        center_lon = map_data[
            "longitude"
        ].mean()

        candidate_layer = pdk.Layer(
            "ScatterplotLayer",
            data=candidate_points,
            get_position=[
                "longitude",
                "latitude",
            ],
            get_radius=1400,
            get_fill_color=[
                37,
                99,
                235,
                170,
            ],
            get_line_color=[
                30,
                64,
                175,
                255,
            ],
            pickable=True,
        )

        existing_layer = pdk.Layer(
            "ScatterplotLayer",
            data=existing_points,
            get_position=[
                "longitude",
                "latitude",
            ],
            get_radius=900,
            get_fill_color=[
                17,
                24,
                39,
                220,
            ],
            get_line_color=[
                17,
                24,
                39,
                255,
            ],
            pickable=True,
        )

        deck = pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=float(center_lat),
                longitude=float(center_lon),
                zoom=11,
                pitch=0,
            ),
            layers=[
                candidate_layer,
                existing_layer,
            ],
            tooltip={
                "text": "{area}\n{type}"
            },
        )

        st.pydeck_chart(
            deck,
            use_container_width=True,
        )

    elif not map_data.empty:

        st.map(
            map_data[
                [
                    "latitude",
                    "longitude",
                ]
            ],
            use_container_width=True,
        )

    # --------------------------------------------------------
    # Location table
    # --------------------------------------------------------

    st.markdown(
        "### Candidate Locations"
    )

    display = candidates[
        [
            "area",
            "latitude",
            "longitude",
            "catchment_radius_km",
        ]
    ].copy()

    display.columns = [
        "Candidate",
        "Latitude",
        "Longitude",
        "Catchment (km)",
    ]

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PAGE 2 — MARKET POTENTIAL
# ============================================================

elif page == "Market Potential":

    st.markdown(
        '<div class="section-title">'
        'Market Potential'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-subtitle">
        Measures the underlying opportunity before considering
        competition, accessibility, rent or cannibalisation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = data["market"]

    if df.empty:

        st.warning(
            "Market Potential data unavailable."
        )

        st.stop()

    ranking = df.sort_values(
        "market_potential_score",
        ascending=False,
    )

    col1, col2, col3 = st.columns(3)

    for index, column in enumerate(
        [col1, col2, col3]
    ):

        if index < len(ranking):

            row = ranking.iloc[index]

            column.metric(
                str(row["area"]),
                f"{row['market_potential_score']:.1f}",
            )

    st.markdown(
        "### Market Opportunity"
    )

    chart_data = (
        ranking[
            [
                "area",
                "market_potential_score",
            ]
        ]
        .set_index("area")
    )

    st.bar_chart(
        chart_data,
        y="market_potential_score",
        use_container_width=True,
    )

    st.markdown(
        "### Driver Breakdown"
    )

    table = ranking[
        [
            "area",
            "population_market_index",
            "purchasing_power_market_index",
            "household_market_index",
            "market_potential_score",
        ]
    ].copy()

    table.columns = [
        "Candidate",
        "Population",
        "Purchasing Power",
        "Households",
        "Market Potential",
    ]

    st.dataframe(
        table.round(1),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Market Potential is a relative 0–100 opportunity "
        "index. It is not a revenue forecast."
    )


# ============================================================
# PAGE 3 — DEMAND PREDICTION
# ============================================================

elif page == "Demand Prediction":

    st.markdown(
        '<div class="section-title">'
        'Demand Prediction'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-subtitle">
        ML-based relative demand signal using public location
        features and synthetic training observations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = data["demand"]

    if df.empty:

        st.warning(
            "Demand Prediction data unavailable."
        )

        st.stop()

    ranking = df.sort_values(
        "predicted_demand_score",
        ascending=False,
    )

    col1, col2, col3 = st.columns(3)

    for index, column in enumerate(
        [col1, col2, col3]
    ):

        if index < len(ranking):

            row = ranking.iloc[index]

            column.metric(
                str(row["area"]),
                f"{row['predicted_demand_score']:.1f}",
            )

    st.markdown(
        "### Predicted Demand"
    )

    chart_data = (
        ranking[
            [
                "area",
                "predicted_demand_score",
            ]
        ]
        .set_index("area")
    )

    st.bar_chart(
        chart_data,
        y="predicted_demand_score",
        use_container_width=True,
    )

    st.markdown(
        "### Model Output"
    )

    table = ranking[
        [
            "area",
            "predicted_demand_index",
            "predicted_demand_score",
            "model_type",
            "training_data",
        ]
    ].copy()

    table.columns = [
        "Candidate",
        "Demand Index",
        "Demand Score",
        "Model",
        "Training Data",
    ]

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "Current training data is synthetic. This validates "
        "the ML mechanism, not Croma-specific predictive "
        "accuracy. Historical retailer sales would replace "
        "the synthetic training set in production."
    )


# ============================================================
# PAGE 4 — DEMAND CAPTURE
# ============================================================

elif page == "Demand Capture":

    st.markdown(
        '<div class="section-title">'
        'Demand Capture'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-subtitle">
        Converts predicted demand into a more realistic
        capturable-demand signal using competition and
        accessibility.
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = data["capture"]

    if df.empty:

        st.warning(
            "Demand Capture data unavailable."
        )

        st.stop()

    ranking = df.sort_values(
        "captured_demand_score",
        ascending=False,
    )

    st.markdown(
        "### Capturable Demand"
    )

    chart_data = ranking[
        [
            "area",
            "captured_demand_score",
        ]
    ].set_index("area")

    st.bar_chart(
        chart_data,
        y="captured_demand_score",
        use_container_width=True,
    )

    st.markdown(
        "### Capture Drivers"
    )

    for _, row in ranking.iterrows():

        with st.container(
            border=True
        ):

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                row["area"],
                f"{row['captured_demand_score']:.1f}",
            )

            c2.metric(
                "Accessibility",
                format_factor(
                    row["accessibility_factor"]
                ),
            )

            c3.metric(
                "Competition",
                format_factor(
                    row["competition_factor"]
                ),
            )

            c4.metric(
                "Captured Demand",
                f"{row['captured_demand_index']:.1f}",
            )

    st.markdown(
        "### Interpretation"
    )

    st.info(
        "Accessibility factors above 1.0 improve the demand "
        "signal; competition factors below 1.0 reduce it. "
        "These are prototype adjustment factors, not sales "
        "probabilities."
    )


# ============================================================
# PAGE 5 — WHAT-IF SIMULATOR
# ============================================================

elif page == "What-if Simulator":

    st.markdown(
        '<div class="section-title">'
        'What-if Simulator'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-subtitle">
        Select a candidate and see how the location performs
        across the major decision fronts.
        </div>
        """,
        unsafe_allow_html=True,
    )

    candidate_names = get_candidate_names(
        data
    )

    selected_candidate = st.selectbox(
        "Select candidate location",
        candidate_names,
        index=(
            candidate_names.index(
                st.session_state.selected_candidate
            )
            if (
                st.session_state.selected_candidate
                in candidate_names
            )
            else 0
        ),
    )

    st.session_state.selected_candidate = (
        selected_candidate
    )

    area_filter = (
        data["features"]["area"]
        .astype(str)
        == selected_candidate
    )

    feature_row = data["features"][
        area_filter
    ]

    market_row = data["market"][
        data["market"]["area"].astype(str)
        == selected_candidate
    ]

    demand_row = data["demand"][
        data["demand"]["area"].astype(str)
        == selected_candidate
    ]

    capture_row = data["capture"][
        data["capture"]["area"].astype(str)
        == selected_candidate
    ]

    economics_row = data["economics"][
        data["economics"]["area"].astype(str)
        == selected_candidate
    ]

    network_row = data["network"][
        data["network"]["area"].astype(str)
        == selected_candidate
    ]

    if (
        market_row.empty
        or demand_row.empty
        or capture_row.empty
        or network_row.empty
    ):

        st.error(
            "Incomplete data for selected candidate."
        )

        st.stop()

    market = market_row.iloc[0]
    demand = demand_row.iloc[0]
    capture = capture_row.iloc[0]
    network = network_row.iloc[0]

    economics = (
        economics_row.iloc[0]
        if not economics_row.empty
        else None
    )

    # --------------------------------------------------------
    # Main score
    # --------------------------------------------------------

    model_rankings = data[
        "recommendation"
    ].sort_values(
        "rank"
    )

    model_winner = model_rankings.iloc[
        0
    ]

    st.markdown(
        f"### Scenario: {selected_candidate}"
    )

    if (
        str(model_winner["area"])
        == selected_candidate
    ):

        st.success(
            "This candidate is currently the model recommendation."
        )

    else:

        st.info(
            f"The model currently recommends "
            f"{model_winner['area']}."
        )

    # --------------------------------------------------------
    # Fronts
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Market Potential",
        safe_round(
            market[
                "market_potential_score"
            ]
        ),
    )

    c2.metric(
        "Predicted Demand",
        safe_round(
            demand[
                "predicted_demand_score"
            ]
        ),
    )

    c3.metric(
        "Demand Capture",
        safe_round(
            capture[
                "captured_demand_score"
            ]
        ),
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Accessibility",
        format_factor(
            capture[
                "accessibility_factor"
            ]
        ),
    )

    c2.metric(
        "Competition",
        format_factor(
            capture[
                "competition_factor"
            ]
        ),
    )

    c3.metric(
        "Network Impact",
        safe_round(
            network[
                "network_impact_score"
            ]
        ),
    )

    # --------------------------------------------------------
    # Economics
    # --------------------------------------------------------

    if economics is not None:

        if (
            economics[
                "economics_data_quality"
            ]
            == "complete"
        ):

            st.metric(
                "Economics",
                safe_round(
                    economics[
                        "economic_attractiveness_score"
                    ]
                ),
            )

        else:

            st.metric(
                "Economics",
                "Pending",
            )

    # --------------------------------------------------------
    # Scenario explanation
    # --------------------------------------------------------

    st.markdown(
        "### Scenario Readout"
    )

    nearest_store = network[
        "nearest_existing_store"
    ]

    nearest_distance = network[
        "nearest_existing_store_distance_km"
    ]

    if pd.notna(
        nearest_store
    ):

        st.markdown(
            f"""
            <div class="insight-box">
            <b>Network exposure:</b>
            nearest fixed store is
            <b>{nearest_store}</b>
            at approximately
            <b>{nearest_distance:.2f} km</b>.<br><br>

            <b>Demand capture:</b>
            {capture['captured_demand_score']:.1f}.<br>

            <b>Network impact:</b>
            {network['network_impact_score']:.1f}.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE 6 — STORE ECONOMICS
# ============================================================

elif page == "Store Economics":

    st.markdown(
        '<div class="section-title">'
        'Store Economics'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-subtitle">
        Demand versus cost attractiveness. Financial metrics
        remain intentionally conservative where public cost data
        is unavailable.
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = data["economics"]

    if df.empty:

        st.warning(
            "Store Economics data unavailable."
        )

        st.stop()

    if (
        "economics_data_quality" in df.columns
        and (
            df["economics_data_quality"]
            != "complete"
        ).all()
    ):

        st.warning(
            "Public rent data has not yet been populated. "
            "No rent values are being fabricated. Economics "
            "therefore remains a partial POC layer."
        )

    ranking = df.sort_values(
        "captured_demand_score",
        ascending=False,
    )

    for _, row in ranking.iterrows():

        with st.container(
            border=True
        ):

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                row["area"],
                safe_round(
                    row[
                        "captured_demand_score"
                    ]
                ),
            )

            rent = row[
                "annual_rent"
            ]

            c2.metric(
                "Annual Rent",
                (
                    "Pending"
                    if pd.isna(rent)
                    else f"€ / ₹ {rent:,.0f}"
                ),
            )

            c3.metric(
                "Rent Attractiveness",
                (
                    "Pending"
                    if pd.isna(
                        row[
                            "rent_attractiveness_score"
                        ]
                    )
                    else safe_round(
                        row[
                            "rent_attractiveness_score"
                        ]
                    )
                ),
            )

            c4.metric(
                "Economic Score",
                (
                    "Partial"
                    if row[
                        "economics_data_quality"
                    ]
                    != "complete"
                    else safe_round(
                        row[
                            "economic_attractiveness_score"
                        ]
                    )
                ),
            )

    st.markdown(
        "### Production Upgrade"
    )

    st.info(
        "With retailer-specific COGS, staffing, utilities, "
        "capex and verified rent, this layer can become a true "
        "profitability / ROI model."
    )


# ============================================================
# PAGE 7 — NETWORK IMPACT
# ============================================================

elif page == "Network Impact":

    st.markdown(
        '<div class="section-title">'
        'Network Impact'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-subtitle">
        Quantifies how much a candidate overlaps with the fixed
        existing network using distance-decay logic.
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = data["network"]

    if df.empty:

        st.warning(
            "Network Impact data unavailable."
        )

        st.stop()

    ranking = df.sort_values(
        "network_impact_score",
        ascending=False,
    )

    chart_data = ranking[
        [
            "area",
            "network_impact_score",
        ]
    ].set_index("area")

    st.bar_chart(
        chart_data,
        y="network_impact_score",
        use_container_width=True,
    )

    st.markdown(
        "### Candidate Network Exposure"
    )

    for _, row in ranking.iterrows():

        with st.container(
            border=True
        ):

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                row["area"],
                safe_round(
                    row[
                        "network_impact_score"
                    ]
                ),
            )

            c2.metric(
                "Nearest Existing",
                str(
                    row[
                        "nearest_existing_store"
                    ]
                    or "—"
                ),
            )

            c3.metric(
                "Distance",
                (
                    "—"
                    if pd.isna(
                        row[
                            "nearest_existing_store_distance_km"
                        ]
                    )
                    else
                    f"{row['nearest_existing_store_distance_km']:.2f} km"
                ),
            )

            c4.metric(
                "Overlap Pressure",
                safe_round(
                    row[
                        "cannibalisation_pressure"
                    ],
                    3,
                ),
            )

    st.markdown(
        "### Method"
    )

    st.info(
        "Overlap uses exponential distance decay against the "
        "fixed existing stores. It is a relative cannibalisation "
        "indicator, not a forecast of transferred sales."
    )


# ============================================================
# PAGE 8 — RECOMMENDATION
# ============================================================

elif page == "Recommendation":

    st.markdown(
        '<div class="section-title">'
        'Recommendation'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-subtitle">
        Final network-aware recommendation and comparison with
        the candidate chosen by management.
        </div>
        """,
        unsafe_allow_html=True,
    )

    recommendation_df = (
        data["recommendation"]
        .sort_values("rank")
    )

    if recommendation_df.empty:

        st.warning(
            "Recommendation data unavailable."
        )

        st.stop()

    model_winner = (
        recommendation_df.iloc[0]
    )

    st.markdown(
        f"""
        <div class="recommendation-box">
            <div style="font-size:13px; opacity:0.7;">
                MODEL RECOMMENDATION
            </div>
            <div class="recommendation-location">
                {model_winner['area']}
            </div>
            <div style="font-size:18px; margin-top:6px;">
                Decision Score:
                <b>{model_winner['decision_score']:.1f}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "### Candidate Ranking"
    )

    ranking_table = recommendation_df[
        [
            "rank",
            "area",
            "decision_score",
            "primary_strength",
            "primary_risk",
            "network_explanation",
        ]
    ].copy()

    ranking_table.columns = [
        "Rank",
        "Candidate",
        "Decision Score",
        "Primary Strength",
        "Primary Risk",
        "Network Context",
    ]

    st.dataframe(
        ranking_table,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### Management What-if"
    )

    candidate_names = (
        recommendation_df[
            "area"
        ]
        .astype(str)
        .tolist()
    )

    selected_candidate = st.selectbox(
        "Which candidate would management choose?",
        candidate_names,
        index=(
            candidate_names.index(
                st.session_state.selected_candidate
            )
            if (
                st.session_state.selected_candidate
                in candidate_names
            )
            else 0
        ),
    )

    st.session_state.selected_candidate = (
        selected_candidate
    )

    # Recalculate recommendation comparison using current
    # network output, without recollecting public data.
    comparison = run_recommendation(
        selected_candidate=selected_candidate
    )[
        "comparison"
    ]

    selected_row = recommendation_df[
        recommendation_df["area"].astype(str)
        == selected_candidate
    ].iloc[0]

    model_col, selected_col = st.columns(2)

    with model_col:

        st.markdown(
            "#### Model Recommendation"
        )

        st.metric(
            "Location",
            model_winner["area"],
        )

        st.metric(
            "Decision Score",
            f"{model_winner['decision_score']:.1f}",
        )

    with selected_col:

        st.markdown(
            "#### Management Selection"
        )

        st.metric(
            "Location",
            selected_candidate,
        )

        st.metric(
            "Decision Score",
            f"{selected_row['decision_score']:.1f}",
        )

    # --------------------------------------------------------
    # Comparison result
    # --------------------------------------------------------

    status = comparison[
        "comparison_status"
    ]

    if (
        status
        == "model_agrees_with_selected"
    ):

        st.success(
            f"Model agrees with management's "
            f"selection of {selected_candidate}."
        )

    elif (
        status
        == "model_prefers_alternative"
    ):

        st.warning(
            f"The model prefers {model_winner['area']} "
            f"over {selected_candidate} by "
            f"{comparison['score_gap']:.1f} points."
        )

    else:

        st.info(
            f"Management's selected candidate "
            f"{selected_candidate} currently "
            f"outperforms the model recommendation."
        )

    st.markdown(
        "### Decision Logic"
    )

    st.markdown(
        """
        The final score combines:

        **15% Market Potential**
        + **15% Demand Prediction**
        + **30% Demand Capture**
        + **15% Store Economics**
        + **25% Network Impact**

        Missing or partial data is excluded and the remaining
        weights are renormalized.
        """
    )

    st.markdown(
        "### Decision Guardrail"
    )

    st.info(
        "This POC is a decision-support system, not an "
        "autonomous store-opening decision. The strongest "
        "production upgrade would be verified retailer sales, "
        "rent and operating-cost data."
    )
