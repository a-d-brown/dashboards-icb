## ── Imports and Config ────────────────────────
import pandas as pd
import numpy as np
import os
import re
from datetime import date
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from chart_utils import plot_icb_bar_chart, plot_line_chart, plot_high_cost_drugs_scatter
from medicine_price_lookup import DmdPriceLookup, OUTPUT_COLUMNS, read_codes_csv, write_results_csv


## ── Styling ────────────────────────────────
pio.templates.default = 'simple_white'
st.set_page_config(page_title="ICB Workstream Dashboard", layout="wide")

# Load all drugs file (ICB / national level)
@st.cache_data
def load_all_drugs(path="All Drugs National.csv"):
    """
    Load All Drugs CSV (cached). Returns None if file missing or load fails.
    """
    try:
        if not os.path.exists(path):
            return None
        return pd.read_csv(path)
    except Exception as e:
        return None

# ── NEW: Load practice-level all drugs file ────────────────────────────────
@st.cache_data
def load_all_drugs_practice(path="All Drugs Practice.parquet"):
    """
    Load the GP-practice-level All Drugs Parquet file (cached).
    Returns None if file missing or load fails.
    Requires: pyarrow  (pip install pyarrow)
    """
    try:
        if not os.path.exists(path):
            return None
        df = pd.read_parquet(path)
        df["Year Month"]    = df["Year Month"].astype(str).str.strip()
        df["ICB plus Code"] = df["ICB plus Code"].astype(str).str.strip()
        return df
    except Exception:
        return None

all_drugs_df          = load_all_drugs()
all_drugs_practice_df = load_all_drugs_practice()   # ← NEW


@st.cache_data
def load_pct_change(path="Spend & Items % Change by BNF and ICB.csv"):
    try:
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        for k in ["ICB plus Code", "BNF Chemical Substance plus Code"]:
            if k in df.columns:
                df[k] = df[k].astype(str).str.strip()
        return df
    except Exception:
        return None

initial_dataset = "Antibacterials"
initial_measure = "Spend per 1000 Patients"
initial_subloc  = "Show all"
initial_highlight = "None"

if "explore_mode" not in st.session_state:
    st.session_state.explore_mode = False

if "current_page" not in st.session_state:
    st.session_state.current_page = "dashboard"


@st.cache_resource(show_spinner="Loading BNF mapping and dm+d extracts…")
def get_price_lookup() -> DmdPriceLookup:
    """Load the local reference data once per Streamlit server."""
    return DmdPriceLookup(Path(__file__).parent / "price_viewer_data")


def get_price_viewer_data_version() -> tuple[str, str]:
    """Return the mapping date and dm+d release version encoded in local filenames."""
    data_dir = Path(__file__).parent / "price_viewer_data"
    mapping_file = next(iter(sorted(data_dir.glob("BNF Snomed Mapping data *.xlsx"))), None)
    ampp_file = next(iter(sorted(data_dir.glob("f_ampp2_*.xml*"))), None)

    mapping_match = re.search(r"(\d{8})", mapping_file.name) if mapping_file else None
    mapping_date = mapping_match.group(1) if mapping_match else "Unknown"
    if mapping_date != "Unknown":
        mapping_date = f"{mapping_date[:4]}-{mapping_date[4:6]}-{mapping_date[6:]}"
    release_match = re.search(r"f_ampp2_([^.]*)", ampp_file.name) if ampp_file else None
    release_version = release_match.group(1) if release_match else "Unknown"
    return mapping_date, release_version

st.markdown("""
    <style>
        .block-container {
            padding-top: 3.75rem;
        }
        .stSelectbox div[data-baseweb="select"] {
            font-size: 18px;
            width: 100%;
            max-width: 800px;
        }
        .st-key-enter_explore button,
        .st-key-enter_price_viewer button,
        .st-key-exit_benchmark_explorer button,
        .st-key-exit_price_viewer button {
            background-color: #e5f2f1;
            border-color: #8dbfbd;
            color: #2f6f73;
        }
        .st-key-enter_explore button:hover,
        .st-key-enter_price_viewer button:hover,
        .st-key-exit_benchmark_explorer button:hover,
        .st-key-exit_price_viewer button:hover {
            background-color: #cfe6e4;
            border-color: #2f6f73;
            color: #24575a;
        }
    </style>
""", unsafe_allow_html=True)

if st.session_state.current_page == "dashboard":

    ## ── Constants: Mappings and Settings ───────────

    sicbl_legend_mapping = {
        '01H': 'Cumbria',
        '84H': 'Durham',
        '13T': 'Newcastle-Gateshead',
        '99C': 'North Tyneside',
        '00L': 'Northumberland',
        '00N': 'South Tyneside',
        '00P': 'Sunderland',
        '16C': 'Tees Valley'
    }

    sub_location_colors = {
        'Cumbria':             '#9467bd',
        'Durham':              '#1f77b4',
        'Newcastle-Gateshead': '#ff7f0e',
        'North Tyneside':      '#e377c2',
        'Northumberland':      '#2ca02c',
        'South Tyneside':      '#bcbd22',
        'Sunderland':          '#d62728',
        'Tees Valley':         '#17becf'
    }

    dataset_measures = {
        "Rifaximin":                       ["Spend per 1000 Patients", "Items per 1000 Patients"],
        "Antibacterials":                  ["Spend per 1000 Patients", "Items per 1000 Patients", "DDD per 1000 Patients"],
        "Gabapentinoids":                  ["Spend per 1000 Patients", "Items per 1000 Patients", "ADQ per 1000 Patients"],
        "Opioids":                         ["Spend per 1000 Patients", "Items per 1000 Patients", "ADQ per 1000 Patients"],
        "Lidocaine Patches":               ["Spend per 1000 Patients", "Items per 1000 Patients"],
        "SABAs":                           ["Spend per 1000 Patients", "Items per 1000 Patients"],
        "Closed Triple Inhalers":          ["Spend per 1000 COPD Patients", "Items per 1000 COPD Patients"],
        "Vitamin B12 IM":                  ["Spend per 1000 Patients", "Items per 1000 Patients"],
        "Vitamin C":                       ["Spend per 1000 Patients", "Items per 1000 Patients"],
        "Vitamin D":                       ["Spend per 1000 Patients", "Items per 1000 Patients"],
        "PPIs":                            ["Spend per 1000 Patients", "Items per 1000 Patients", "ADQ per 1000 Patients"],
        "Bath & Shower Emollients":        ["Spend per 1000 Patients", "Items per 1000 Patients"],
        "Dicycloverine":                   ["Spend per 1000 Patients", "Items per 1000 Patients"],
        "Promethazine":                    ["Spend per 1000 Patients", "Items per 1000 Patients"],
        "Enteral Nutrition":               ["Spend per 1000 Patients", "Items per 1000 Patients"],
        "Dressings":                       ["Spend per 1000 Patients", "Items per 1000 Patients"],
        "Cloral betaine + Chloral hydrate":["Spend per 1000 Patients", "Items per 1000 Patients"],
        "Specials":                        ["Spend per 1000 Patients", "Items per 1000 Patients"],
        "High Cost Drugs":                 ["Spend per 1000 Patients", "Items per 1000 Patients"]
    }

    measure_metadata = {
        "Spend per 1000 Patients": {
            "denominator_column": "List Size",
            "numerator_column":   "Actual Cost",
            "prefix": "£"
        },
        "Items per 1000 Patients": {
            "denominator_column": "List Size",
            "numerator_column":   "Items",
            "prefix": ""
        },
        "ADQ per 1000 Patients": {
            "denominator_column": "List Size",
            "numerator_column":   "ADQ Usage",
            "prefix": ""
        },
        "DDD per 1000 Patients": {
            "denominator_column": "List Size",
            "numerator_column":   "DDD Usage",
            "prefix": ""
        },
        "Spend per 1000 COPD Patients": {
            "denominator_column": "COPD List Size",
            "numerator_column":   "Actual Cost",
            "prefix": "£"
        },
        "Items per 1000 COPD Patients": {
            "denominator_column": "COPD List Size",
            "numerator_column":   "Items",
            "prefix": ""
        },
        "Δ from previous": {"prefix": "", "suffix": "%"},
        "Δ from last year": {"prefix": "", "suffix": "%"}
    }

    NATIONAL_COPD_LIST_SIZE = 1175163

    ## ── Functions: Preprocessing and Loading ───────

    def preprocess_prescribing_data(df, is_national, mapping=None):
        if not is_national:
            df = df[df['PCN'] != 'DUMMY']
            df = df[~df['Practice'].str.contains(r'\( ?[CD] ?\d', na=False)]
            df['Practice'] = df['Practice'].str.rstrip(',').str.title()
            df['Commissioner / Provider Code'] = df['Commissioner / Provider Code'].str.slice(0, -2)
            df['sub_location'] = df['Commissioner / Provider Code'].map(mapping)

        df.rename(columns={'Year Month': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m')
        df['formatted_date'] = df['date'].dt.strftime('%b %Y')
        df['date_period'] = df['date'].dt.to_period('M').dt.to_timestamp()

        latest_15_months = df['date_period'].drop_duplicates().sort_values(ascending=False).head(15).tolist()
        recent_3m    = latest_15_months[:3]
        prev_3m      = latest_15_months[3:6]
        last_year_3m = latest_15_months[12:15]

        def label_period(period):
            if period in recent_3m:    return "recent"
            elif period in prev_3m:    return "previous"
            elif period in last_year_3m: return "last_year"
            else:                       return "other"

        df['period_tag'] = df['date_period'].apply(label_period)

        for col in ['ADQ Usage', 'DDD Usage']:
            if col not in df.columns:
                df[col] = 0

        return df

    def aggregate_substance_data(df, group_cols):
        summary = df.groupby(group_cols, as_index=False)[['Items', 'Actual Cost', 'ADQ Usage', 'DDD Usage']].sum().round(1)
        base = df.drop_duplicates(subset=group_cols).drop(columns=['Items', 'Actual Cost', 'ADQ Usage', 'DDD Usage', 'BNF Chemical Substance plus Code'])
        merged = pd.merge(base, summary, on=group_cols)
        merged['BNF Chemical Substance plus Code'] = 'Drugs Aggregated'
        return merged

    def compute_period_aggregates(df, period_tag, dataset_type="Prescribing", group_cols=('Practice',), numerators=None, denominators=None):
        if numerators is None or denominators is None:
            if dataset_type == "High Cost Drugs":
                numerators   = ['Actual Cost', 'Items']
                denominators = ['List Size']
            else:
                numerators   = ['Actual Cost', 'Items', 'ADQ Usage', 'DDD Usage']
                denominators = ['List Size', 'COPD List Size']
        subset   = df[df['period_tag'] == period_tag]
        sum_df   = subset.groupby(list(group_cols), as_index=False)[numerators].sum().round(1)
        mean_df  = subset.groupby(list(group_cols), as_index=False)[denominators].mean().round(1)
        agg      = pd.merge(sum_df, mean_df, on=list(group_cols))
        return agg

    @st.cache_data
    def load_data(dataset_type):
        icb_path        = f"__{dataset_type} - ICB Dashboard.csv"
        icb_data_raw    = pd.read_csv(icb_path)

        if dataset_type == "High Cost Drugs":
            icb_data_preprocessed   = icb_data_raw
            national_data_preprocessed = None
        else:
            icb_data_preprocessed = preprocess_prescribing_data(
                icb_data_raw, is_national=False, mapping=sicbl_legend_mapping
            )
            national_path      = f"__{dataset_type} - ICB Dashboard NATIONAL.csv"
            national_data_raw  = pd.read_csv(national_path)
            national_data_preprocessed = preprocess_prescribing_data(
                national_data_raw, is_national=True
            )

        return icb_data_preprocessed, national_data_preprocessed

    copd_list_size_df = pd.read_csv("copd_register_may25.csv")
    copd_list_size_df['Practice'] = copd_list_size_df['Practice'].str.title()
    copd_list_size_df = copd_list_size_df[['Practice', 'COPD Register']]
    copd_list_size_df.rename(columns={'COPD Register': 'COPD List Size'}, inplace=True)

    # Header and page navigation
    header_title, header_navigation = st.columns([3, 2])

    with header_title:
        st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: flex-start; margin-bottom: 0.75rem;">
                <div style="font-size: clamp(20px, 4vw, 36px); font-weight: bold;">
                    NENC Medicines Optimisation Workstream Dashboard
                </div>
                <div style="background-color: #ffc107; color: black; font-size: 0.9em; font-weight: bold; padding: 2px 8px; border-radius: 4px; margin-top: 0.3em;">
                    BETA
                </div>
            </div>
        """, unsafe_allow_html=True)

    with header_navigation:
        def enter_explore():
            st.session_state.current_page = "bnf_explorer"

        def enter_price_viewer():
            st.session_state.current_page = "price_viewer"

        st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

        bnf_button_col, price_button_col = st.columns(2, gap="small")
        with bnf_button_col:
            st.button("Benchmark Explorer", on_click=enter_explore, key="enter_explore")
        with price_button_col:
            st.button("Price Viewer", on_click=enter_price_viewer, key="enter_price_viewer")

    st.markdown("<hr style='margin: 0.25rem 0 1rem;'>", unsafe_allow_html=True)

    # Dashboard selectors
    col_chart_controls, col2_subloc, col_divider, col3_dataset = st.columns([2.5, 1.4, 0.05, 1.4])

    with col2_subloc:
        subloc_options = ["Show all"] + list(sub_location_colors.keys())
        selected_sublocation = st.selectbox(
            "Select Sub-location:",
            options=subloc_options,
            index=subloc_options.index(initial_subloc) if initial_subloc in subloc_options else 0,
            key="subloc_selector"
        )
        if selected_sublocation == "Show all":
            selected_sublocations = list(sub_location_colors.keys())
        else:
            selected_sublocations = [selected_sublocation]

    with col_divider:
        st.markdown(
            """<div style="min-height: 160px; border-left: 1px solid #ddd; margin: 0 8px;"></div>""",
            unsafe_allow_html=True
        )

    with col3_dataset:
        dataset_type = st.selectbox(
            "Select Dataset:",
            options=list(dataset_measures.keys()),
            index=list(dataset_measures.keys()).index(initial_dataset)
        )

        measure_options = dataset_measures.get(dataset_type, [])
        default_measure_index = (
            measure_options.index(initial_measure)
            if initial_measure in measure_options else 0
        )

        measure_type = st.selectbox(
            "Select Measure:",
            options=measure_options,
            index=default_measure_index
        )

        if dataset_type == "High Cost Drugs":
            cost_filter_threshold = st.slider(
                "Minimum Cost per Item to include:",
                min_value=25.0, max_value=2500.0, value=25.0, step=25.0,
                format="£%0.0f"
            )
        else:
            cost_filter_threshold = None

    ## ── Data Loading ───────────────────────────────
    mode_option = "Current Rate"
    with col_chart_controls:
        if dataset_type != "High Cost Drugs":
            mode_option = st.radio(
                "Bar chart display:",
                ["Current Rate", "Recent change", "Annual change"],
                index=0,
                horizontal=True,
            )

    icb_data_preprocessed, national_data_preprocessed = load_data(dataset_type)

    numerator_column   = measure_metadata[measure_type]["numerator_column"]
    denominator_column = measure_metadata[measure_type]["denominator_column"]

    early_recent_practices = (
        icb_data_preprocessed[icb_data_preprocessed["period_tag"] == "recent"]
        .groupby("Practice", as_index=False)
        .agg({"sub_location": "first"})
    )

    with col2_subloc:
        selected_practice = None

        if selected_sublocation != "Show all":
            recent_active = icb_data_preprocessed[
                (icb_data_preprocessed["period_tag"] == "recent") &
                (icb_data_preprocessed[numerator_column] > 0) &
                (icb_data_preprocessed["sub_location"].isin(selected_sublocations))
            ]
            recent_practices = recent_active["Practice"].dropna().sort_values().unique()
            practice_options = ["None"] + list(recent_practices)
            default_index = (
                practice_options.index(initial_highlight)
                if initial_highlight in practice_options else 0
            )
            selected_practice_option = st.selectbox(
                "Select Practice:",
                options=practice_options,
                index=default_index
            )
            if selected_practice_option != "None":
                selected_practice = selected_practice_option

    if dataset_type == "High Cost Drugs":
        all_hcd_data   = icb_data_preprocessed.copy()
        total_practices = all_hcd_data["Practice"].nunique()
        threshold       = total_practices * 0.5

        hcd_practice_counts = (
            all_hcd_data.groupby("BNF Presentation plus Code")["Practice"]
            .nunique().reset_index()
            .rename(columns={"Practice": "num_practices"})
        )
        low_distribution_drugs = hcd_practice_counts[
            hcd_practice_counts["num_practices"] < threshold
        ]["BNF Presentation plus Code"]

        icb_data_preprocessed = icb_data_preprocessed[
            icb_data_preprocessed["BNF Presentation plus Code"].isin(low_distribution_drugs)
        ]

        if cost_filter_threshold is not None:
            if selected_practice:
                filtered_subset = icb_data_preprocessed[icb_data_preprocessed["Practice"] == selected_practice]
            elif selected_sublocation != "Show all":
                filtered_subset = icb_data_preprocessed[icb_data_preprocessed["sub_location"] == selected_sublocation]
            else:
                filtered_subset = icb_data_preprocessed.copy()

            avg_cost_per_drug = (
                filtered_subset
                .groupby("BNF Presentation plus Code")[["Actual Cost", "Items"]].sum()
                .assign(avg_cost_per_item=lambda df: df["Actual Cost"] / df["Items"])
            )
            high_cost_drugs = avg_cost_per_drug[
                avg_cost_per_drug["avg_cost_per_item"] >= cost_filter_threshold
            ].index
            high_cost_drug_data_preprocessed = icb_data_preprocessed[
                icb_data_preprocessed["BNF Presentation plus Code"].isin(high_cost_drugs)
            ]

        if cost_filter_threshold is not None:
            icb_data_preprocessed = icb_data_preprocessed[
                icb_data_preprocessed["3m Average Cost per Item"] >= cost_filter_threshold
            ]

    if dataset_type != "High Cost Drugs":
        icb_data_preprocessed = icb_data_preprocessed.merge(
            copd_list_size_df, on='Practice', how='left'
        )
        if measure_metadata[measure_type]["denominator_column"] == "COPD List Size":
            icb_data_preprocessed = icb_data_preprocessed[
                icb_data_preprocessed["COPD List Size"].notna()
            ]

    ## ── Aggregation and Measure Calculation ─────────

    icb_data_aggregated = aggregate_substance_data(
        icb_data_preprocessed, ['Practice', 'date_period']
    )

    if 'period_tag' in icb_data_preprocessed.columns:
        period_tags = icb_data_preprocessed[['Practice', 'date_period', 'period_tag']].drop_duplicates()
        icb_data_aggregated = icb_data_aggregated.merge(
            period_tags, on=['Practice', 'date_period'], how='left', suffixes=('', '_drop')
        )
        icb_data_aggregated.drop(
            columns=[col for col in icb_data_aggregated.columns if col.endswith('_drop')],
            inplace=True
        )

    if national_data_preprocessed is not None:
        national_data_raw_merged = aggregate_substance_data(national_data_preprocessed, ['date'])
        if denominator_column == "COPD List Size":
            national_data_raw_merged[measure_type] = (
                (national_data_raw_merged[numerator_column] / NATIONAL_COPD_LIST_SIZE) * 1000
            ).round(1)
        else:
            national_data_raw_merged[measure_type] = (
                (national_data_raw_merged[numerator_column] / national_data_raw_merged[denominator_column]) * 1000
            ).round(1)
    else:
        national_data_raw_merged = None

    means = compute_period_aggregates(icb_data_aggregated, 'recent', dataset_type)

    def make_3m_window_label(date_period_series, start_index):
        periods = (
            pd.Series(date_period_series.drop_duplicates())
            .sort_values(ascending=False).tolist()
        )
        window = periods[start_index:start_index + 3]
        if len(window) == 0:
            return ""
        latest_ts   = pd.to_datetime(window[0])
        earliest_ts = pd.to_datetime(window[-1])
        if len(window) == 1:
            return earliest_ts.strftime("%b %Y")
        if earliest_ts.year == latest_ts.year:
            return f"{earliest_ts.strftime('%b')}–{latest_ts.strftime('%b %Y')}"
        else:
            return f"{earliest_ts.strftime('%b %Y')}–{latest_ts.strftime('%b %Y')}"

    date_series       = icb_data_preprocessed["date_period"]
    current_3m_label  = make_3m_window_label(date_series, 0)

    if dataset_type != "High Cost Drugs":
        previous_3m_label  = make_3m_window_label(date_series, 3)
        last_year_3m_label = make_3m_window_label(date_series, 12)

    columns_to_drop = [col for col in [
        'Actual Cost', 'Items', 'ADQ Usage', 'DDD Usage',
        'List Size', 'COPD List Size', 'date', 'formatted_date'
    ] if col in icb_data_aggregated.columns]
    base_means = icb_data_aggregated.drop_duplicates(subset='Practice').drop(columns=columns_to_drop)

    icb_means_merged = pd.merge(base_means, means, on='Practice')
    icb_means_merged[measure_type] = (
        (icb_means_merged[numerator_column] / icb_means_merged[denominator_column]) * 1000
    ).round(1)

    if dataset_type != "High Cost Drugs":
        previous_means = compute_period_aggregates(
            icb_data_aggregated, 'previous', dataset_type, group_cols=('Practice',)
        )
        icb_means_merged = pd.merge(
            icb_means_merged, previous_means, on='Practice', suffixes=('', '_prev')
        )
        icb_means_merged[f"{measure_type} (previous)"] = (
            icb_means_merged[f"{numerator_column}_prev"] /
            icb_means_merged[f"{denominator_column}_prev"] * 1000
        ).round(1)
        icb_means_merged["Δ from previous"] = (
            (icb_means_merged[measure_type] - icb_means_merged[f"{measure_type} (previous)"]) /
            icb_means_merged[f"{measure_type} (previous)"] * 100
        ).round(1)

    if dataset_type != "High Cost Drugs":
        last_year_means = compute_period_aggregates(
            icb_data_aggregated, 'last_year', dataset_type, group_cols=('Practice',)
        )
        icb_means_merged = pd.merge(
            icb_means_merged, last_year_means, on='Practice', suffixes=('', '_lastyear')
        )
        icb_means_merged[f"{measure_type} (last year)"] = (
            icb_means_merged[f"{numerator_column}_lastyear"] /
            icb_means_merged[f"{denominator_column}_lastyear"] * 1000
        ).round(1)
        icb_means_merged["Δ from last year"] = (
            (icb_means_merged[measure_type] - icb_means_merged[f"{measure_type} (last year)"]) /
            icb_means_merged[f"{measure_type} (last year)"] * 100
        ).round(1)

    icb_means_merged['Items'] = icb_means_merged['Items'].round(0).astype(int)
    icb_means_merged.rename(columns={'Items': 'Items (monthly average)'}, inplace=True)
    icb_means_merged['Items (monthly average)'] = icb_means_merged['Items (monthly average)'] / 3

    total_numerator   = means[numerator_column].sum()
    total_denominator = means[denominator_column].sum()
    icb_average_value = round((total_numerator / total_denominator) * 1000, 1) if total_denominator else 0.0

    icb_data_aggregated[measure_type] = (
        (icb_data_aggregated[numerator_column] / icb_data_aggregated[denominator_column]) * 1000
    ).round(1)

    ### STREAMLIT LAYOUT ------------

    if selected_sublocation == "Show all":
        selected_sublocations = list(sub_location_colors.keys())
    else:
        selected_sublocations = [selected_sublocation]

    use_delta_chart = False
    use_year_change = False

    if dataset_type != "High Cost Drugs":
        use_delta_chart = mode_option == "Recent change"
        use_year_change = mode_option == "Annual change"

        if use_delta_chart:
            header_title = f'Practice comparisons: {dataset_type} - {measure_type} ({current_3m_label} vs {previous_3m_label})'
        elif use_year_change:
            header_title = f'Practice comparisons: {dataset_type} - {measure_type} ({current_3m_label} vs {last_year_3m_label})'
        else:
            header_title = f'Practice comparisons: {dataset_type} - {measure_type} ({current_3m_label})'
    else:
        header_title = f'Practice comparisons: {dataset_type} - {measure_type} ({current_3m_label})'

    st.header(header_title)

    filtered_data = icb_means_merged[
        (icb_means_merged['sub_location'].isin(selected_sublocations)) &
        (icb_means_merged[measure_type] > 0)
    ]

    filtered_colors = {k: v for k, v in sub_location_colors.items() if k in selected_sublocations}
    if len(selected_sublocations) > 1 and filtered_colors:
        legend_cols = st.columns(len(filtered_colors))
        for i, (subloc, color) in enumerate(filtered_colors.items()):
            with legend_cols[i]:
                st.markdown(
                    f"<div style='display: flex; align-items: center;'>"
                    f"<div style='width: 14px; height: 14px; background-color: {color}; margin-right: 6px; border: 1px solid #00000022;'></div>"
                    f"<span style='font-size: 13px;'>{subloc}</span></div>",
                    unsafe_allow_html=True
                )

    if use_delta_chart:
        y_axis_column = "Δ from previous"
        y_axis_label  = f"{dataset_type} {measure_type} Δ from previous 3m"
    elif use_year_change:
        y_axis_column = "Δ from last year"
        y_axis_label  = f"{dataset_type} {measure_type} Δ from same 3m last year"
    else:
        y_axis_column = measure_type
        y_axis_label  = f"{dataset_type} {measure_type}"

    delta_column = None
    if dataset_type != "High Cost Drugs":
        if use_delta_chart:
            delta_column = "Δ from previous"
        elif use_year_change:
            delta_column = "Δ from last year"

    bar_fig = plot_icb_bar_chart(
        filtered_data=filtered_data,
        measure_type=y_axis_column,
        sub_location_colors=sub_location_colors,
        icb_average_value=icb_average_value,
        dataset_type=dataset_type,
        measure_metadata=measure_metadata,
        selected_practice=selected_practice,
        delta_column=delta_column,
        y_axis_label=y_axis_label
    )
    st.plotly_chart(bar_fig, use_container_width=True)
    st.markdown("---")

    # ── Trend / Scatter Chart Section ─────────────────────────
    if dataset_type == "High Cost Drugs":
        if selected_practice:
            st.header(f"Top 100 High Cost Drugs (latest 3m) at {selected_practice}")
            view_mode = st.radio(
                "Select view", options=["View as table", "View as scatterplot"],
                horizontal=True, label_visibility="collapsed"
            )
            practice_data = high_cost_drug_data_preprocessed[
                high_cost_drug_data_preprocessed["Practice"] == selected_practice
            ].copy()
            top_presentations = (
                practice_data.groupby("BNF Presentation plus Code", as_index=False)[["Items", "Actual Cost"]]
                .sum().sort_values("Actual Cost", ascending=False)
                .head(100)["BNF Presentation plus Code"].sort_values().tolist()
            )
            if view_mode == "View as scatterplot":
                top100_summary = (
                    practice_data[practice_data["BNF Presentation plus Code"].isin(top_presentations)]
                    .groupby("BNF Presentation plus Code", as_index=False)[["Items", "Actual Cost"]]
                    .sum().sort_values("Actual Cost", ascending=False)
                )
                scatter_fig = plot_high_cost_drugs_scatter(top100_summary)
                if scatter_fig:
                    st.plotly_chart(scatter_fig, use_container_width=True)
                else:
                    st.info("No data available for the selected practice.")
            elif view_mode == "View as table":
                top_drugs_grouped = (
                    practice_data[practice_data["BNF Presentation plus Code"].isin(top_presentations)]
                    .groupby("BNF Presentation plus Code", as_index=False)[["Actual Cost", "Items"]].sum()
                )
                top_drugs_grouped["Average Cost per Item"] = top_drugs_grouped["Actual Cost"] / top_drugs_grouped["Items"]
                top_drugs_table = top_drugs_grouped.sort_values("Actual Cost", ascending=False)
                styled_table = top_drugs_table.style \
                    .format({"Actual Cost": "£{:,.0f}", "Items": "{:,.0f}", "Average Cost per Item": "£{:,.0f}"}) \
                    .background_gradient(subset=["Average Cost per Item"], cmap="Reds")
                st.dataframe(styled_table, use_container_width=True, hide_index=True)

        elif selected_sublocation and selected_sublocation != "Show all":
            st.header(f"Top 100 High Cost Drugs (latest 3m) in {selected_sublocation}")
            view_mode = st.radio(
                "Select view", options=["View as table", "View as scatterplot"],
                horizontal=True, label_visibility="collapsed"
            )
            subloc_data = high_cost_drug_data_preprocessed[
                high_cost_drug_data_preprocessed["sub_location"].str.strip().str.casefold() ==
                selected_sublocation.strip().casefold()
            ].copy()
            top_presentations = (
                subloc_data.groupby("BNF Presentation plus Code", as_index=False)[["Items", "Actual Cost"]]
                .sum().sort_values("Actual Cost", ascending=False)
                .head(100)["BNF Presentation plus Code"].sort_values().tolist()
            )
            if view_mode == "View as scatterplot":
                top100_summary = (
                    subloc_data[subloc_data["BNF Presentation plus Code"].isin(top_presentations)]
                    .groupby("BNF Presentation plus Code", as_index=False)[["Items", "Actual Cost"]]
                    .sum().sort_values("Actual Cost", ascending=False)
                )
                scatter_fig = plot_high_cost_drugs_scatter(top100_summary)
                if scatter_fig:
                    st.plotly_chart(scatter_fig, use_container_width=True)
                else:
                    st.info("No data available for the selected sublocation.")
            elif view_mode == "View as table":
                top_drugs_grouped = (
                    subloc_data[subloc_data["BNF Presentation plus Code"].isin(top_presentations)]
                    .groupby("BNF Presentation plus Code", as_index=False)[["Actual Cost", "Items"]].sum()
                )
                top_drugs_grouped["Average Cost per Item"] = top_drugs_grouped["Actual Cost"] / top_drugs_grouped["Items"]
                top_drugs_table = top_drugs_grouped.sort_values("Actual Cost", ascending=False)
                styled_table = top_drugs_table.style \
                    .format({"Actual Cost": "£{:,.0f}", "Items": "{:,.0f}", "Average Cost per Item": "£{:,.0f}"}) \
                    .background_gradient(subset=["Average Cost per Item"], cmap="Reds")
                st.dataframe(styled_table, use_container_width=True, hide_index=True)
            selected_presentation = st.selectbox("Select a drug to view practice-level breakdown:", top_presentations)
            if selected_presentation:
                subset = subloc_data[subloc_data["BNF Presentation plus Code"] == selected_presentation]
                num_practices = subset["Practice"].nunique()
                st.subheader(f"{num_practices} practices in {selected_sublocation} prescribe {selected_presentation}")
                practice_breakdown = (
                    subset.groupby("Practice", as_index=False)[["Actual Cost", "Items"]]
                    .sum().sort_values("Actual Cost", ascending=False)
                )
                styled_practice_breakdown = practice_breakdown.style \
                    .format({"Actual Cost": "£{:,.0f}", "Items": "{:,.0f}"})
                st.dataframe(styled_practice_breakdown, use_container_width=True, hide_index=True)

        elif selected_sublocation and selected_sublocation == "Show all":
            st.header(f"Top 100 High Cost Drugs (latest 3m) across NENC")
            view_mode = st.radio(
                "Select view", options=["View as table", "View as scatterplot"],
                horizontal=True, label_visibility="collapsed"
            )
            top_presentations = (
                high_cost_drug_data_preprocessed
                .groupby("BNF Presentation plus Code", as_index=False)[["Items", "Actual Cost"]]
                .sum().sort_values("Actual Cost", ascending=False)
                .head(100)["BNF Presentation plus Code"].sort_values().tolist()
            )
            if view_mode == "View as scatterplot":
                top100_summary = (
                    high_cost_drug_data_preprocessed[
                        high_cost_drug_data_preprocessed["BNF Presentation plus Code"].isin(top_presentations)
                    ].groupby("BNF Presentation plus Code", as_index=False)[["Items", "Actual Cost"]]
                    .sum().sort_values("Actual Cost", ascending=False)
                )
                scatter_fig = plot_high_cost_drugs_scatter(top100_summary)
                if scatter_fig:
                    st.plotly_chart(scatter_fig, use_container_width=True)
                else:
                    st.info("No data available for the selected sublocation.")
            elif view_mode == "View as table":
                top_drugs_grouped = (
                    high_cost_drug_data_preprocessed[
                        high_cost_drug_data_preprocessed["BNF Presentation plus Code"].isin(top_presentations)
                    ].groupby("BNF Presentation plus Code", as_index=False)[["Actual Cost", "Items"]].sum()
                )
                top_drugs_grouped["Average Cost per Item"] = top_drugs_grouped["Actual Cost"] / top_drugs_grouped["Items"]
                top_drugs_table = top_drugs_grouped.sort_values("Actual Cost", ascending=False)
                styled_table = top_drugs_table.style \
                    .format({"Actual Cost": "£{:,.0f}", "Items": "{:,.0f}", "Average Cost per Item": "£{:,.0f}"}) \
                    .background_gradient(subset=["Average Cost per Item"], cmap="Reds")
                st.dataframe(styled_table, use_container_width=True, hide_index=True)
            selected_presentation = st.selectbox("Select a drug to view practice-level breakdown:", top_presentations)
            if selected_presentation:
                subset = high_cost_drug_data_preprocessed[
                    high_cost_drug_data_preprocessed["BNF Presentation plus Code"] == selected_presentation
                ]
                num_practices = subset["Practice"].nunique()
                st.subheader(f"{num_practices} practices in NENC prescribe {selected_presentation}")
                practice_breakdown = (
                    subset.groupby("Practice", as_index=False)[["sub_location", "Actual Cost", "Items"]]
                    .sum().sort_values("Actual Cost", ascending=False)
                )
                practice_breakdown_display = practice_breakdown.rename(columns={"sub_location": "Sub-location"})
                styled_practice_breakdown = practice_breakdown_display.style \
                    .format({"Actual Cost": "£{:,.0f}", "Items": "{:,.0f}"}) \
                    .set_properties(**{"text-align": "left"})
                st.dataframe(styled_practice_breakdown, use_container_width=True, hide_index=True)

    if dataset_type == "Specials":
        df = icb_data_preprocessed.copy()
        presentation_col = next(
            (c for c in ["BNF Presentation plus Code", "Presentation", "BNF Presentation"] if c in df.columns),
            None,
        )
        if "Actual Cost" not in df.columns or "Items" not in df.columns or presentation_col is None:
            st.info("Specials dataset missing expected columns (Presentation / Actual Cost / Items).")
        else:
            recent_df = df[df["period_tag"] == "recent"].copy()
            if selected_practice:
                practice_data  = recent_df[recent_df["Practice"] == selected_practice].copy()
                context_label  = f"at {selected_practice}"
            elif selected_sublocation and selected_sublocation != "Show all":
                practice_data  = recent_df[recent_df["sub_location"].str.strip().str.casefold() == selected_sublocation.strip().casefold()].copy()
                context_label  = f"in {selected_sublocation}"
            else:
                practice_data  = recent_df.copy()
                context_label  = "across NENC"

            st.header(f"Top 100 Specials ({current_3m_label}) {context_label}")
            top_presentations = (
                practice_data
                .groupby(presentation_col, as_index=False)[["Items", "Actual Cost"]]
                .sum().sort_values("Actual Cost", ascending=False)
                .head(100)[presentation_col].sort_values().tolist()
            )
            if len(top_presentations) == 0:
                st.info("No specials data available for the selected scope.")
            else:
                top_drugs_grouped = (
                    practice_data[practice_data[presentation_col].isin(top_presentations)]
                    .groupby(presentation_col, as_index=False)[["Actual Cost", "Items"]].sum()
                )
                top_drugs_grouped["Average Cost per Item"] = top_drugs_grouped.apply(
                    lambda r: (r["Actual Cost"] / r["Items"]) if r["Items"] else 0, axis=1
                )
                top_drugs_table = top_drugs_grouped.sort_values("Actual Cost", ascending=False)
                styled_table = top_drugs_table.style \
                    .format({"Actual Cost": "£{:,.0f}", "Items": "{:,.0f}", "Average Cost per Item": "£{:,.0f}"}) \
                    .background_gradient(subset=["Average Cost per Item"], cmap="Reds")
                st.dataframe(styled_table, use_container_width=True, hide_index=True)

                selected_presentation = st.selectbox(
                    f"Select a special to view practice-level breakdown:", top_presentations
                )
                if selected_presentation:
                    subset = recent_df[recent_df[presentation_col] == selected_presentation]
                    num_practices = subset["Practice"].nunique()
                    st.subheader(f"{num_practices} practices in NENC prescribe {selected_presentation}")
                    practice_breakdown = (
                        subset.groupby("Practice", as_index=False)[["Actual Cost", "Items"]]
                        .sum().sort_values("Actual Cost", ascending=False)
                    )
                    practice_breakdown["Average Cost per Item"] = practice_breakdown.apply(
                        lambda r: (r["Actual Cost"] / r["Items"]) if r["Items"] else 0, axis=1
                    )
                    styled_practice_breakdown = practice_breakdown.style \
                        .format({"Actual Cost": "£{:,.0f}", "Items": "{:,.0f}", "Average Cost per Item": "£{:,.0f}"})
                    st.dataframe(styled_practice_breakdown, use_container_width=True, hide_index=True)

    if dataset_type != "High Cost Drugs":
        st.header(f'Trend analysis: {dataset_type} - {measure_type}')
        if len(selected_sublocations) > 1 and filtered_colors:
            legend_cols = st.columns(len(filtered_colors))
            for i, (subloc, color) in enumerate(filtered_colors.items()):
                with legend_cols[i]:
                    st.markdown(
                        f"<div style='display: flex; align-items: center;'>"
                        f"<div style='width: 14px; height: 14px; background-color: {color}; margin-right: 6px; border: 1px solid #00000022;'></div>"
                        f"<span style='font-size: 13px;'>{subloc}</span></div>",
                        unsafe_allow_html=True
                    )
        if selected_practice:
            line_fig = plot_line_chart(
                icb_data_aggregated, national_data_raw_merged,
                sub_location=selected_sublocation, selected_sublocation=selected_sublocation,
                selected_practice=selected_practice, measure_type=measure_type,
                dataset_type=dataset_type, mode="practice"
            )
        else:
            line_fig = plot_line_chart(
                icb_data_aggregated, national_data_raw_merged,
                sub_location=None, selected_sublocation=selected_sublocation,
                selected_practice=None, measure_type=measure_type,
                dataset_type=dataset_type, mode="sublocations",
                sub_location_colors=sub_location_colors
            )
        st.plotly_chart(line_fig, use_container_width=True)


elif st.session_state.current_page == "bnf_explorer":
    # ══════════════════════════════════════════════════════════════════════
    # BNF EXPLORER MODE
    # ══════════════════════════════════════════════════════════════════════

    # ── NEW: View-level radio buttons ─────────────────────────────────────
    explorer_content, col_bnf = st.columns([2, 1])
    with explorer_content:
        if st.button("← Go to prebuilt measures", key="exit_benchmark_explorer"):
            st.session_state.current_page = "dashboard"
            st.rerun()

        st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)

        view_level = st.radio(
            "View data by:",
            ["ICB level", "GP Practice level"],
            index=0,
            horizontal=True,
            help=(
                "ICB level: all practices within each ICB aggregated into a single row.  "
                "GP Practice level: one row per GP practice (NENC only)."
            )
        )

    # ── Load national data (always needed for ICB mode and England benchmark) ──
    all_drugs_df = load_all_drugs()
    if all_drugs_df is None:
        st.warning("All Drugs National.csv not found.")
        st.stop()

    all_drugs_df = all_drugs_df.copy()
    all_drugs_df["ICB plus Code"] = all_drugs_df["ICB plus Code"].astype(str).str.strip()
    all_drugs_df["Year Month"]    = all_drugs_df["Year Month"].astype(str).str.strip()

    # Detect national file months
    nat_available_months = sorted(all_drugs_df["Year Month"].dropna().unique())
    if len(nat_available_months) < 2:
        st.warning("Expected two months in All Drugs National.csv — found fewer.")
        st.stop()

    nat_month_from = nat_available_months[0]
    nat_month_to   = nat_available_months[-1]

    # ICB-mode month labels come from the national file
    month_from       = nat_month_from
    month_to         = nat_month_to
    month_from_label = pd.to_datetime(month_from, format="%Y%m").strftime("%b %Y")
    month_to_label   = pd.to_datetime(month_to,   format="%Y%m").strftime("%b %Y")

    # ── BNF level constants ────────────────────────────────────────────────
    BNF_LEVEL_ORDER = [
        "BNF Chapter",
        "BNF Section",
        "BNF Sub Paragraph",
        "BNF Chemical Substance",
    ]
    BNF_LEVELS = {
        "BNF Chapter":            "BNF Chapter plus Code",
        "BNF Section":            "BNF Section plus Code",
        "BNF Sub Paragraph":      "BNF Sub Paragraph plus Code",
        "BNF Chemical Substance": "BNF Chemical Substance plus Code",
    }
    DEFAULT_ICB = "NHS NORTH EAST AND NORTH CUMBRIA INTEGRATED CARE BOARD (QHM)"
    NENC_ICB    = DEFAULT_ICB   # alias used in practice mode

    # ── UI columns: ICB selector + BNF search ─────────────────────────────
    selected_icb = NENC_ICB

    with col_bnf:
        search_level_label = st.selectbox(
            "Filter by BNF category:",
            BNF_LEVEL_ORDER,
            index=0,
        )
        search_col = BNF_LEVELS[search_level_label]

        # Initialise — overwritten below depending on the selected level
        bnf_search            = ""
        selected_term         = None
        breakdown_level_label = search_level_label

        if search_level_label == "BNF Chapter":
            # ── Chapter mode: dropdown of all available chapters ─────────────
            # There are only ~20 chapters so a dropdown is clean and avoids the
            # need for free-text search at this level.
            _ch = all_drugs_df[search_col].dropna().astype(str).drop_duplicates()
            chapter_options = (
                pd.DataFrame({
                    "name": _ch,
                    "num":  _ch.str.extract(r'\((\d+)\)\s*$')[0].astype(float)
                })
                .sort_values("num")["name"]
                .tolist()
            )
            chosen_chapter = st.selectbox(
                "Select BNF Chapter:",
                ["-- Select a chapter --"] + chapter_options,
                key="bnf_chapter_select",
            )
            if chosen_chapter != "-- Select a chapter --":
                selected_term = chosen_chapter
                breakdown_level_label = st.selectbox(
                    "Select breakdown granularity",
                    BNF_LEVEL_ORDER,   # all levels are valid when filtering from Chapter
                    index=0,
                    key="chapter_breakdown_granularity",
                )

        else:
            # ── Other levels: free-text search (existing behaviour) ───────────
            # Too many items (hundreds to thousands) for a dropdown, so keep
            # the search-and-suggest pattern.
            bnf_search = st.text_input(
                f"Search within {search_level_label}:",
                placeholder=f"Type to find a specific {search_level_label.lower()}...",
                key="bnf_search_box",
                label_visibility="collapsed"
            )

            all_terms = (
                all_drugs_df[search_col]
                .dropna().astype(str).drop_duplicates().sort_values().tolist()
            )

            if bnf_search.strip():
                term        = bnf_search.strip().casefold()
                suggestions = [x for x in all_terms if term in x.casefold()][:10]

                if suggestions and search_level_label != "BNF Chemical Substance":
                    st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
                    category_col, granularity_col = st.columns(2, gap="small")
                    allowed_breakdown_levels = BNF_LEVEL_ORDER[BNF_LEVEL_ORDER.index(search_level_label):]

                    with category_col:
                        chosen_suggestion = st.selectbox(
                            "Select category to breakdown",
                            ["-- No breakdown --"] + suggestions,
                            key=f"bnf_suggestion_box_{search_level_label}",
                        )
                        if chosen_suggestion != "-- No breakdown --":
                            selected_term = chosen_suggestion

                    with granularity_col:
                        breakdown_level_label = st.selectbox(
                            "Select breakdown granularity",
                            allowed_breakdown_levels,
                            index=0,
                        )

    breakdown_col = BNF_LEVELS[breakdown_level_label]

    # Resolve which BNF column and label to use for the table's first column
    search_text       = bnf_search.strip()
    display_bnf_col   = search_col
    display_bnf_label = search_level_label
    if selected_term:
        display_bnf_col   = breakdown_col
        display_bnf_label = breakdown_level_label

    # ── Shared helper functions ────────────────────────────────────────────

    def build_pivot(df, bnf_col, p_month_from, p_month_to):
        """
        Aggregates df to (bnf_col × month) and pivots to wide format.
        Returns derived rate columns: spend/items/spend-per-item for from/to months.
        """
        for col in ["Actual Cost", "Items", "List Size"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        agg = (
            df.groupby([bnf_col, "Year Month"], as_index=False)
            .agg({"Actual Cost": "sum", "Items": "sum", "List Size": "sum"})
        )
        pivot = agg.pivot_table(
            index=bnf_col,
            columns="Year Month",
            values=["Actual Cost", "Items", "List Size"],
            aggfunc="first"
        )
        pivot.columns = [f"{val}_{col}" for val, col in pivot.columns]
        pivot = pivot.reset_index()

        for col in [
            f"Actual Cost_{p_month_from}", f"Actual Cost_{p_month_to}",
            f"Items_{p_month_from}",       f"Items_{p_month_to}",
            f"List Size_{p_month_from}",   f"List Size_{p_month_to}",
        ]:
            if col not in pivot.columns:
                pivot[col] = 0.0

        pivot[[
            f"Actual Cost_{p_month_from}", f"Actual Cost_{p_month_to}",
            f"Items_{p_month_from}",       f"Items_{p_month_to}",
            f"List Size_{p_month_from}",   f"List Size_{p_month_to}",
        ]] = pivot[[
            f"Actual Cost_{p_month_from}", f"Actual Cost_{p_month_to}",
            f"Items_{p_month_from}",       f"Items_{p_month_to}",
            f"List Size_{p_month_from}",   f"List Size_{p_month_to}",
        ]].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        pivot["spend_per_patient_from"] = np.where(pivot[f"List Size_{p_month_from}"] == 0, 0.0, pivot[f"Actual Cost_{p_month_from}"] / pivot[f"List Size_{p_month_from}"])
        pivot["spend_per_patient_to"]   = np.where(pivot[f"List Size_{p_month_to}"]   == 0, 0.0, pivot[f"Actual Cost_{p_month_to}"]   / pivot[f"List Size_{p_month_to}"])
        pivot["items_per_patient_from"] = np.where(pivot[f"List Size_{p_month_from}"] == 0, 0.0, pivot[f"Items_{p_month_from}"]        / pivot[f"List Size_{p_month_from}"])
        pivot["items_per_patient_to"]   = np.where(pivot[f"List Size_{p_month_to}"]   == 0, 0.0, pivot[f"Items_{p_month_to}"]          / pivot[f"List Size_{p_month_to}"])
        pivot["spend_per_item_from"]    = np.where(pivot[f"Items_{p_month_from}"]     == 0, 0.0, pivot[f"Actual Cost_{p_month_from}"]  / pivot[f"Items_{p_month_from}"])
        pivot["spend_per_item_to"]      = np.where(pivot[f"Items_{p_month_to}"]       == 0, 0.0, pivot[f"Actual Cost_{p_month_to}"]    / pivot[f"Items_{p_month_to}"])

        return pivot

    # ── NEW: practice-level pivot (includes Practice + Practice Code in index) ──
    def build_pivot_practice(df, bnf_col, p_month_from, p_month_to, id_cols):
        """
        Like build_pivot but adds id_cols (e.g. Practice, Practice Code) to the
        pivot index so each row represents one practice × BNF category.
        """
        for col in ["Actual Cost", "Items", "List Size"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        index_cols = id_cols + [bnf_col]

        agg = (
            df.groupby(index_cols + ["Year Month"], as_index=False)
            .agg({"Actual Cost": "sum", "Items": "sum", "List Size": "sum"})
        )
        pivot = agg.pivot_table(
            index=index_cols,
            columns="Year Month",
            values=["Actual Cost", "Items", "List Size"],
            aggfunc="first"
        )
        pivot.columns = [f"{val}_{col}" for val, col in pivot.columns]
        pivot = pivot.reset_index()

        for col in [
            f"Actual Cost_{p_month_from}", f"Actual Cost_{p_month_to}",
            f"Items_{p_month_from}",       f"Items_{p_month_to}",
            f"List Size_{p_month_from}",   f"List Size_{p_month_to}",
        ]:
            if col not in pivot.columns:
                pivot[col] = 0.0

        pivot[[
            f"Actual Cost_{p_month_from}", f"Actual Cost_{p_month_to}",
            f"Items_{p_month_from}",       f"Items_{p_month_to}",
            f"List Size_{p_month_from}",   f"List Size_{p_month_to}",
        ]] = pivot[[
            f"Actual Cost_{p_month_from}", f"Actual Cost_{p_month_to}",
            f"Items_{p_month_from}",       f"Items_{p_month_to}",
            f"List Size_{p_month_from}",   f"List Size_{p_month_to}",
        ]].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        pivot["spend_per_patient_from"] = np.where(pivot[f"List Size_{p_month_from}"] == 0, 0.0, pivot[f"Actual Cost_{p_month_from}"] / pivot[f"List Size_{p_month_from}"])
        pivot["spend_per_patient_to"]   = np.where(pivot[f"List Size_{p_month_to}"]   == 0, 0.0, pivot[f"Actual Cost_{p_month_to}"]   / pivot[f"List Size_{p_month_to}"])
        pivot["items_per_patient_from"] = np.where(pivot[f"List Size_{p_month_from}"] == 0, 0.0, pivot[f"Items_{p_month_from}"]        / pivot[f"List Size_{p_month_from}"])
        pivot["items_per_patient_to"]   = np.where(pivot[f"List Size_{p_month_to}"]   == 0, 0.0, pivot[f"Items_{p_month_to}"]          / pivot[f"List Size_{p_month_to}"])
        pivot["spend_per_item_from"]    = np.where(pivot[f"Items_{p_month_from}"]     == 0, 0.0, pivot[f"Actual Cost_{p_month_from}"]  / pivot[f"Items_{p_month_from}"])
        pivot["spend_per_item_to"]      = np.where(pivot[f"Items_{p_month_to}"]       == 0, 0.0, pivot[f"Actual Cost_{p_month_to}"]    / pivot[f"Items_{p_month_to}"])

        return pivot

    def pct_change_arrays(f_arr, t_arr):
        """
        Numeric % change between two rate arrays.
          - both 0        → 0.0
          - 0 → positive  → 999999.0 (new — no prior baseline)
          - positive → 0  → -100.0   (discontinued)
          - positive → positive → normal % change rounded to 1 dp
        """
        f   = f_arr.astype(float)
        t   = t_arr.astype(float)
        num = np.full(len(f), np.nan, dtype=float)
        num[(f == 0) & (t == 0)] = 0.0
        num[(f == 0) & (t  > 0)] = 999999.0
        num[(f  > 0) & (t == 0)] = -100.0
        mask = (f > 0) & (t > 0)
        num[mask] = np.round((t[mask] - f[mask]) / f[mask] * 100.0, 1)
        return num

    def pct_diff_vs_national(icb_arr, nat_arr):
        """% difference of a rate vs England rate. NaN where national == 0."""
        icb = icb_arr.astype(float)
        nat = nat_arr.astype(float)
        num = np.full(len(icb), np.nan, dtype=float)
        mask = nat > 0
        num[mask] = np.round((icb[mask] - nat[mask]) / nat[mask] * 100.0, 1)
        return num

    def colour_signed(val):
        """Red for positive, green for negative, neutral for zero/NaN/sentinel."""
        if pd.isna(val) or val == 999999.0:
            return ""
        try:
            val = float(val)
        except Exception:
            return ""
        if val > 0:   return "background-color: #f4cccc;"
        elif val < 0: return "background-color: #d9ead3;"
        return ""

    # ── Shared column config (second-level MultiIndex keys) ───────────────
    shared_column_config = {
        "Actual Cost":                  st.column_config.NumberColumn(format="£%.0f"),
        "Cost per Item":                st.column_config.NumberColumn(format="£%.2f"),
        "Items":                        st.column_config.NumberColumn(format="%d"),
        "Financial Impact for 1 month": st.column_config.NumberColumn(format="£%.0f"),
        "Spend per Patient":            st.column_config.NumberColumn(format="%.1f%%"),
        "Items per Patient":            st.column_config.NumberColumn(format="%.1f%%"),
        "Spend per Item":               st.column_config.NumberColumn(format="%.1f%%"),
    }

    # ══════════════════════════════════════════════════════════════════════
    # BRANCH: ICB level (original behaviour)
    # ══════════════════════════════════════════════════════════════════════
    st.header("Benchmark Explorer")

    if view_level == "ICB level":

        # Filter to selected ICB and apply BNF search
        icb_df = all_drugs_df[all_drugs_df["ICB plus Code"] == selected_icb].copy()

        if selected_term:
            icb_df = icb_df[icb_df[search_col].astype(str) == selected_term].copy()
            # display_bnf_col / label already set to breakdown level above
        elif search_text:
            icb_df = icb_df[
                icb_df[search_col].astype(str).str.contains(search_text, case=False, na=False)
            ].copy()

        if display_bnf_col not in icb_df.columns:
            st.error(f"Missing {display_bnf_col} column.")
            st.stop()

        pivot = build_pivot(icb_df, display_bnf_col, month_from, month_to)

        # England benchmark pivot
        eng_df    = all_drugs_df[all_drugs_df["ICB plus Code"] == "England"].copy()
        eng_pivot = build_pivot(eng_df, display_bnf_col, month_from, month_to)
        eng_pivot = eng_pivot[[
            display_bnf_col,
            "spend_per_patient_to", "items_per_patient_to", "spend_per_item_to"
        ]].rename(columns={
            "spend_per_patient_to": "eng_spend_per_patient_to",
            "items_per_patient_to": "eng_items_per_patient_to",
            "spend_per_item_to":    "eng_spend_per_item_to",
        })
        pivot = pivot.merge(eng_pivot, on=display_bnf_col, how="left")

        # Rename display column
        pivot = pivot.rename(columns={display_bnf_col: display_bnf_label})
        display_bnf_col = display_bnf_label  # updated reference

        for col in ["eng_spend_per_patient_to", "eng_items_per_patient_to", "eng_spend_per_item_to"]:
            pivot[col] = pivot[col].fillna(0.0)

        # % change over time
        pivot["Spend per Patient - % Change"] = pct_change_arrays(pivot["spend_per_patient_from"].to_numpy(), pivot["spend_per_patient_to"].to_numpy())
        pivot["Items per Patient - % Change"] = pct_change_arrays(pivot["items_per_patient_from"].to_numpy(), pivot["items_per_patient_to"].to_numpy())
        pivot["Spend per Item - % Change"]    = pct_change_arrays(pivot["spend_per_item_from"].to_numpy(),    pivot["spend_per_item_to"].to_numpy())

        # % vs national
        pivot["Spend per Patient - % vs National"] = pct_diff_vs_national(pivot["spend_per_patient_to"].to_numpy(), pivot["eng_spend_per_patient_to"].to_numpy())
        pivot["Items per Patient - % vs National"] = pct_diff_vs_national(pivot["items_per_patient_to"].to_numpy(), pivot["eng_items_per_patient_to"].to_numpy())
        pivot["Spend per Item - % vs National"]    = pct_diff_vs_national(pivot["spend_per_item_to"].to_numpy(),    pivot["eng_spend_per_item_to"].to_numpy())

        # Financial impact columns
        pivot["Financial Impact of Change in Spend per Patient"] = np.where(
            (pivot["Spend per Patient - % Change"] == 999999.0) | pivot["Spend per Patient - % Change"].isna(),
            np.nan,
            (pivot["Spend per Patient - % Change"] / 100.0) * pivot[f"Actual Cost_{month_from}"]
        ).round(0)

        pivot["Financial Impact of Difference from National Spend per Patient"] = np.where(
            pivot["eng_spend_per_patient_to"] == 0,
            np.nan,
            pivot[f"Actual Cost_{month_to}"] - (pivot["eng_spend_per_patient_to"] * pivot[f"List Size_{month_to}"])
        ).round(0)

        # Build display table
        table_df = pivot[[
            display_bnf_col,
            f"Actual Cost_{month_to}",
            f"Items_{month_to}",
            "spend_per_item_to",
            "Spend per Patient - % Change",
            "Financial Impact of Change in Spend per Patient",
            "Items per Patient - % Change",
            "Spend per Item - % Change",
            "Spend per Patient - % vs National",
            "Financial Impact of Difference from National Spend per Patient",
            "Items per Patient - % vs National",
            "Spend per Item - % vs National",
        ]].copy().rename(columns={
            f"Actual Cost_{month_to}":                                       "Actual Cost",
            f"Items_{month_to}":                                             "Items",
            "spend_per_item_to":                                             "Cost per Item",
            "Financial Impact of Change in Spend per Patient":               "Financial Impact for 1 month - Change",
            "Financial Impact of Difference from National Spend per Patient":"Financial Impact for 1 month - National",
        })

        table_df["Actual Cost"]                              = table_df["Actual Cost"].round(0)
        table_df["Items"]                                    = table_df["Items"].round(0).astype("Int64")
        table_df["Cost per Item"]                            = table_df["Cost per Item"].round(1)
        table_df["Spend per Patient - % Change"]             = table_df["Spend per Patient - % Change"].round(1)
        table_df["Financial Impact for 1 month - Change"]   = table_df["Financial Impact for 1 month - Change"].round(0)
        table_df["Items per Patient - % Change"]             = table_df["Items per Patient - % Change"].round(1)
        table_df["Spend per Item - % Change"]                = table_df["Spend per Item - % Change"].round(1)
        table_df["Spend per Patient - % vs National"]        = table_df["Spend per Patient - % vs National"].round(1)
        table_df["Financial Impact for 1 month - National"]  = table_df["Financial Impact for 1 month - National"].round(0)
        table_df["Items per Patient - % vs National"]        = table_df["Items per Patient - % vs National"].round(1)
        table_df["Spend per Item - % vs National"]           = table_df["Spend per Item - % vs National"].round(1)

        table_df = table_df[(table_df["Items"] > 0) & (table_df["Actual Cost"] >= 30)].reset_index(drop=True)

        table_df.columns = pd.MultiIndex.from_tuples([
            ("",                                                              display_bnf_label),
            (f"Current values ({month_to_label})",                           "Actual Cost"),
            (f"Current values ({month_to_label})",                           "Items"),
            (f"Current values ({month_to_label})",                           "Cost per Item"),
            (f"Change over time ({month_from_label} \u2192 {month_to_label})", "Spend per Patient"),
            (f"Change over time ({month_from_label} \u2192 {month_to_label})", "Financial Impact for 1 month"),
            (f"Change over time ({month_from_label} \u2192 {month_to_label})", "Items per Patient"),
            (f"Change over time ({month_from_label} \u2192 {month_to_label})", "Spend per Item"),
            (f"Difference relative to national ({month_to_label})",           "Spend per Patient"),
            (f"Difference relative to national ({month_to_label})",           "Financial Impact for 1 month"),
            (f"Difference relative to national ({month_to_label})",           "Items per Patient"),
            (f"Difference relative to national ({month_to_label})",           "Spend per Item"),
        ])

        signed_cols = [
            (f"Change over time ({month_from_label} \u2192 {month_to_label})", "Spend per Patient"),
            (f"Change over time ({month_from_label} \u2192 {month_to_label})", "Financial Impact for 1 month"),
            (f"Change over time ({month_from_label} \u2192 {month_to_label})", "Items per Patient"),
            (f"Change over time ({month_from_label} \u2192 {month_to_label})", "Spend per Item"),
            (f"Difference relative to national ({month_to_label})",            "Spend per Patient"),
            (f"Difference relative to national ({month_to_label})",            "Financial Impact for 1 month"),
            (f"Difference relative to national ({month_to_label})",            "Items per Patient"),
            (f"Difference relative to national ({month_to_label})",            "Spend per Item"),
        ]

        styled_table = table_df.style.applymap(colour_signed, subset=signed_cols)

        row_height    = 48
        header_height = 60
        table_height  = max(0, min(900, header_height + len(table_df) * row_height))

        st.dataframe(
            styled_table,
            use_container_width=True,
            hide_index=True,
            column_config=shared_column_config,
            height=table_height,
        )

        csv = table_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download table (CSV)", csv, file_name="exploration_table.csv")


    # ══════════════════════════════════════════════════════════════════════
    # BRANCH: GP Practice level (NEW)
    # ══════════════════════════════════════════════════════════════════════
    else:

        if all_drugs_practice_df is None:
            st.warning(
                "All Drugs Practice.csv not found. "
                "Please ensure this file is present in the dashboard directory."
            )
            st.stop()

        # Filter to NENC practices only (exclude any England/other ICB rows)
        practice_df = all_drugs_practice_df[
            all_drugs_practice_df["ICB plus Code"] == NENC_ICB
        ].copy()

        if practice_df.empty:
            st.warning(f"No practice-level data found for ICB: {NENC_ICB}")
            st.stop()

        # Detect months from practice file
        prac_available_months = sorted(practice_df["Year Month"].dropna().unique())
        if len(prac_available_months) < 2:
            st.warning("Expected two months in All Drugs Practice.csv — found fewer.")
            st.stop()

        prac_month_from = prac_available_months[0]
        prac_month_to   = prac_available_months[-1]
        prac_month_from_label = pd.to_datetime(prac_month_from, format="%Y%m").strftime("%b %Y")
        prac_month_to_label   = pd.to_datetime(prac_month_to,   format="%Y%m").strftime("%b %Y")

        # ── Gate: a BNF filter must be applied before running the practice pivot ─
        # Without one the result is ~230K rows — slow to compute and impossible
        # to read. The gate message adapts to whichever input mode is active.
        if not selected_term and not search_text:
            if search_level_label == "BNF Chapter":
                st.info(
                    "📂 **Select a BNF Chapter** from the dropdown above to view "
                    "GP practice data."
                )
            else:
                st.info(
                    "🔍 **Search for a BNF category** above to view GP practice data.  \n"
                    f"Type in the search box to find a specific {search_level_label.lower()}."
                )
            st.stop()

        # Apply BNF search / breakdown filter to practice data
        if selected_term:
            practice_df = practice_df[
                practice_df[search_col].astype(str) == selected_term
            ].copy()
        elif search_text:
            practice_df = practice_df[
                practice_df[search_col].astype(str).str.contains(search_text, case=False, na=False)
            ].copy()

        if display_bnf_col not in practice_df.columns:
            st.error(f"Missing column '{display_bnf_col}' in practice data.")
            st.stop()

        # Build practice pivot (Practice + Practice Code + BNF col as index)
        prac_pivot = build_pivot_practice(
            practice_df, display_bnf_col,
            prac_month_from, prac_month_to,
            id_cols=["Practice", "Practice Code"]
        )

        # ── England benchmark from national file ──────────────────────────
        # nat_month_to (202603) is one month before prac_month_to (202604) — acceptable proxy.
        eng_df    = all_drugs_df[all_drugs_df["ICB plus Code"] == "England"].copy()
        eng_pivot = build_pivot(eng_df, display_bnf_col, nat_month_from, nat_month_to)
        eng_pivot = eng_pivot[[
            display_bnf_col,
            "spend_per_patient_to", "items_per_patient_to", "spend_per_item_to"
        ]].rename(columns={
            "spend_per_patient_to": "eng_spend_per_patient_to",
            "items_per_patient_to": "eng_items_per_patient_to",
            "spend_per_item_to":    "eng_spend_per_item_to",
        })

        prac_pivot = prac_pivot.merge(eng_pivot, on=display_bnf_col, how="left")

        # Rename BNF display column
        prac_pivot = prac_pivot.rename(columns={display_bnf_col: display_bnf_label})

        for col in ["eng_spend_per_patient_to", "eng_items_per_patient_to", "eng_spend_per_item_to"]:
            prac_pivot[col] = prac_pivot[col].fillna(0.0)

        # % change over time
        prac_pivot["Spend per Patient - % Change"] = pct_change_arrays(prac_pivot["spend_per_patient_from"].to_numpy(), prac_pivot["spend_per_patient_to"].to_numpy())
        prac_pivot["Items per Patient - % Change"] = pct_change_arrays(prac_pivot["items_per_patient_from"].to_numpy(), prac_pivot["items_per_patient_to"].to_numpy())
        prac_pivot["Spend per Item - % Change"]    = pct_change_arrays(prac_pivot["spend_per_item_from"].to_numpy(),    prac_pivot["spend_per_item_to"].to_numpy())

        # % vs national (England benchmark)
        prac_pivot["Spend per Patient - % vs National"] = pct_diff_vs_national(prac_pivot["spend_per_patient_to"].to_numpy(), prac_pivot["eng_spend_per_patient_to"].to_numpy())
        prac_pivot["Items per Patient - % vs National"] = pct_diff_vs_national(prac_pivot["items_per_patient_to"].to_numpy(), prac_pivot["eng_items_per_patient_to"].to_numpy())
        prac_pivot["Spend per Item - % vs National"]    = pct_diff_vs_national(prac_pivot["spend_per_item_to"].to_numpy(),    prac_pivot["eng_spend_per_item_to"].to_numpy())

        # Financial impact: change over time
        prac_pivot["Financial Impact of Change in Spend per Patient"] = np.where(
            (prac_pivot["Spend per Patient - % Change"] == 999999.0) | prac_pivot["Spend per Patient - % Change"].isna(),
            np.nan,
            (prac_pivot["Spend per Patient - % Change"] / 100.0) * prac_pivot[f"Actual Cost_{prac_month_from}"]
        ).round(0)

        # Financial impact: difference from national
        prac_pivot["Financial Impact of Difference from National Spend per Patient"] = np.where(
            prac_pivot["eng_spend_per_patient_to"] == 0,
            np.nan,
            prac_pivot[f"Actual Cost_{prac_month_to}"] - (
                prac_pivot["eng_spend_per_patient_to"] * prac_pivot[f"List Size_{prac_month_to}"]
            )
        ).round(0)

        # Build display table — same columns as ICB mode, plus "GP Practice"
        prac_table_df = prac_pivot[[
            display_bnf_label,
            "Practice",                    # becomes "GP Practice"
            f"Actual Cost_{prac_month_to}",
            f"Items_{prac_month_to}",
            "spend_per_item_to",
            "Spend per Patient - % Change",
            "Financial Impact of Change in Spend per Patient",
            "Items per Patient - % Change",
            "Spend per Item - % Change",
            "Spend per Patient - % vs National",
            "Financial Impact of Difference from National Spend per Patient",
            "Items per Patient - % vs National",
            "Spend per Item - % vs National",
        ]].copy().rename(columns={
            f"Actual Cost_{prac_month_to}":                                        "Actual Cost",
            f"Items_{prac_month_to}":                                              "Items",
            "spend_per_item_to":                                                   "Cost per Item",
            "Practice":                                                            "GP Practice",
            "Financial Impact of Change in Spend per Patient":                     "Financial Impact for 1 month - Change",
            "Financial Impact of Difference from National Spend per Patient":      "Financial Impact for 1 month - National",
        })

        prac_table_df["Actual Cost"]                              = prac_table_df["Actual Cost"].round(0)
        prac_table_df["Items"]                                    = prac_table_df["Items"].round(0).astype("Int64")
        prac_table_df["Cost per Item"]                            = prac_table_df["Cost per Item"].round(1)
        prac_table_df["Spend per Patient - % Change"]             = prac_table_df["Spend per Patient - % Change"].round(1)
        prac_table_df["Financial Impact for 1 month - Change"]   = prac_table_df["Financial Impact for 1 month - Change"].round(0)
        prac_table_df["Items per Patient - % Change"]             = prac_table_df["Items per Patient - % Change"].round(1)
        prac_table_df["Spend per Item - % Change"]                = prac_table_df["Spend per Item - % Change"].round(1)
        prac_table_df["Spend per Patient - % vs National"]        = prac_table_df["Spend per Patient - % vs National"].round(1)
        prac_table_df["Financial Impact for 1 month - National"]  = prac_table_df["Financial Impact for 1 month - National"].round(0)
        prac_table_df["Items per Patient - % vs National"]        = prac_table_df["Items per Patient - % vs National"].round(1)
        prac_table_df["Spend per Item - % vs National"]           = prac_table_df["Spend per Item - % vs National"].round(1)

        prac_table_df = prac_table_df[
            (prac_table_df["Items"] > 0) & (prac_table_df["Actual Cost"] >= 30)
        ].reset_index(drop=True)

        # MultiIndex column headers — identical structure to ICB mode, with GP Practice added
        prac_table_df.columns = pd.MultiIndex.from_tuples([
            ("",                                                                    display_bnf_label),
            ("",                                                                    "GP Practice"),
            (f"Current values ({prac_month_to_label})",                            "Actual Cost"),
            (f"Current values ({prac_month_to_label})",                            "Items"),
            (f"Current values ({prac_month_to_label})",                            "Cost per Item"),
            (f"Change over time ({prac_month_from_label} \u2192 {prac_month_to_label})", "Spend per Patient"),
            (f"Change over time ({prac_month_from_label} \u2192 {prac_month_to_label})", "Financial Impact for 1 month"),
            (f"Change over time ({prac_month_from_label} \u2192 {prac_month_to_label})", "Items per Patient"),
            (f"Change over time ({prac_month_from_label} \u2192 {prac_month_to_label})", "Spend per Item"),
            (f"Difference relative to national ({prac_month_to_label})",           "Spend per Patient"),
            (f"Difference relative to national ({prac_month_to_label})",           "Financial Impact for 1 month"),
            (f"Difference relative to national ({prac_month_to_label})",           "Items per Patient"),
            (f"Difference relative to national ({prac_month_to_label})",           "Spend per Item"),
        ])

        prac_signed_cols = [
            (f"Change over time ({prac_month_from_label} \u2192 {prac_month_to_label})", "Spend per Patient"),
            (f"Change over time ({prac_month_from_label} \u2192 {prac_month_to_label})", "Financial Impact for 1 month"),
            (f"Change over time ({prac_month_from_label} \u2192 {prac_month_to_label})", "Items per Patient"),
            (f"Change over time ({prac_month_from_label} \u2192 {prac_month_to_label})", "Spend per Item"),
            (f"Difference relative to national ({prac_month_to_label})",                 "Spend per Patient"),
            (f"Difference relative to national ({prac_month_to_label})",                 "Financial Impact for 1 month"),
            (f"Difference relative to national ({prac_month_to_label})",                 "Items per Patient"),
            (f"Difference relative to national ({prac_month_to_label})",                 "Spend per Item"),
        ]

        prac_styled_table = prac_table_df.style.applymap(colour_signed, subset=prac_signed_cols)

        row_height    = 48
        header_height = 60
        prac_table_height = max(0, min(900, header_height + len(prac_table_df) * row_height))

        st.dataframe(
            prac_styled_table,
            use_container_width=True,
            hide_index=True,
            column_config=shared_column_config,
            height=prac_table_height,
        )

        csv = prac_table_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download table (CSV)", csv,
            file_name="practice_exploration_table.csv"
        )


elif st.session_state.current_page == "price_viewer":
    if st.button("← Go to prebuilt measures", key="exit_price_viewer"):
        st.session_state.current_page = "dashboard"
        st.rerun()

    st.title("BNF medicine price lookup")
    st.caption(
        "Look up pack-level dm+d prices using the local BNF-to-SNOMED mapping "
        "and reference-data extracts."
    )
    uploaded = st.file_uploader(
        "Upload a CSV with a BNF Code or bnf_code column",
        type="csv",
        key="price_viewer_upload",
    )
    as_of = st.date_input(
        "Prices effective on or before",
        value=date.today(),
        key="price_viewer_as_of",
    )

    if uploaded:
        try:
            codes = read_codes_csv(uploaded.getvalue())
            if not codes:
                st.error("The CSV contains no data rows.")
            else:
                rows = get_price_lookup().lookup_codes(codes, as_of)
                st.success(f"Processed {len(codes)} input code(s); returned {len(rows)} row(s).")
                st.dataframe(
                    rows,
                    use_container_width=True,
                    hide_index=True,
                    column_order=OUTPUT_COLUMNS,
                )
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                    write_results_csv(rows, tmp.name)
                    csv_bytes = Path(tmp.name).read_bytes()
                Path(tmp.name).unlink(missing_ok=True)
                st.download_button(
                    "Download results CSV",
                    csv_bytes,
                    "bnf_medicine_prices.csv",
                    "text/csv",
                    key="price_viewer_download",
                )
        except (ValueError, FileNotFoundError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.exception(exc)

    mapping_date, release_version = get_price_viewer_data_version()
    st.caption(
        "Contains NHSBSA Dictionary of Medicines and Devices (dm+d) data, licensed under "
        "[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). "
        f"Source: NHS TRUD. BNF mapping date: {mapping_date}; dm+d release: {release_version}."
    )
