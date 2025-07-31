## ── Imports and Config ────────────────────────
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from chart_utils import plot_icb_bar_chart, plot_line_chart, plot_high_cost_drugs_scatter


# Experimental deeplinking
ENABLE_DEEPLINKING = False

if ENABLE_DEEPLINKING:
    if "query_updated" not in st.session_state:
        st.session_state.query_updated = False
    else:
        st.session_state.query_updated = False  # Reset flag on each rerun

    query_params = st.query_params

    initial_dataset = query_params.get("dataset", "Antibacterials")
    initial_measure = query_params.get("measure", "Spend per 1000 Patients")
    initial_subloc = query_params.get("sublocation", "Show all")
    initial_highlight = query_params.get("highlight", "None")
else:
    initial_dataset = "Antibacterials"
    initial_measure = "Spend per 1000 Patients"
    initial_subloc = "Show all"
    initial_highlight = "None"


## ── Styling ────────────────────────────────
pio.templates.default = 'simple_white'
st.set_page_config(page_title="ICB Workstream Dashboard", layout="wide")

st.markdown("""
    <style>
        .stSelectbox div[data-baseweb="select"] {
            font-size: 18px;
            width: 100%;
            max-width: 800px;
        }
    </style>
""", unsafe_allow_html=True)

## ── Constants: Mappings and Settings ───────────

# Add mapping for local SICBL names
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

# Define fixed colors for sub_locations
sub_location_colors = {
    'Cumbria': '#9467bd',
    'Durham': '#1f77b4',
    'Newcastle-Gateshead': '#ff7f0e',
    'North Tyneside': '#e377c2',
    'Northumberland': '#2ca02c',
    'South Tyneside': '#bcbd22',
    'Sunderland': '#d62728',
    'Tees Valley': '#17becf'
}

# Define available measures per dataset - dataset name must match filename
dataset_measures = {
    "Antibacterials": ["Spend per 1000 Patients", "Items per 1000 Patients", "DDD per 1000 Patients"],
    "Gabapentinoids": ["Spend per 1000 Patients", "Items per 1000 Patients", "ADQ per 1000 Patients"],
    "Opioids": ["Spend per 1000 Patients", "Items per 1000 Patients", "ADQ per 1000 Patients"],
    "Lidocaine Patches": ["Spend per 1000 Patients", "Items per 1000 Patients"],
    "SABAs": ["Spend per 1000 Patients", "Items per 1000 Patients"],
    "Closed Triple Inhalers": ["Spend per 1000 COPD Patients", "Items per 1000 COPD Patients"],
    "Vitamin D": ["Spend per 1000 Patients", "Items per 1000 Patients"],
    "PPIs": ["Spend per 1000 Patients", "Items per 1000 Patients", "ADQ per 1000 Patients"],
    "Bath & Shower Emollients": ["Spend per 1000 Patients", "Items per 1000 Patients"],
    "High Cost Drugs": ["Spend per 1000 Patients", "Items per 1000 Patients"]
}

# Measure Metadata ─────────────────────────────
measure_metadata = {
    "Spend per 1000 Patients": {
        "denominator_column": "List Size",
        "numerator_column": "Actual Cost",
        "prefix": "£"
    },
    "Items per 1000 Patients": {
        "denominator_column": "List Size",
        "numerator_column": "Items",
        "prefix": ""
    },
    "ADQ per 1000 Patients": {
        "denominator_column": "List Size",
        "numerator_column": "ADQ Usage",
        "prefix": ""
    },
    "DDD per 1000 Patients": {
        "denominator_column": "List Size",
        "numerator_column": "DDD Usage",
        "prefix": ""
    },
    "Spend per 1000 COPD Patients": {
        "denominator_column": "COPD List Size",
        "numerator_column": "Actual Cost",
        "prefix": "£"
    },
    "Items per 1000 COPD Patients": {
        "denominator_column": "COPD List Size",
        "numerator_column": "Items",
        "prefix": ""
    },
    "Δ from previous": {
    "prefix": "",
    "suffix": "%"
    },
    "Δ from last year": {
    "prefix": "",
    "suffix": "%"
    }

}


NATIONAL_COPD_LIST_SIZE = 1175163


## ── Functions: Preprocessing and Loading ───────

# Preprocess data to standardised format
def preprocess_prescribing_data(df, is_national, mapping=None):
    # Clean and map only for local ICB data
    if not is_national:
        df = df[df['PCN'] != 'DUMMY']
        df = df[~df['Practice'].str.contains(r'\( ?[CD] ?\d', na=False)]
        df['Practice'] = df['Practice'].str.rstrip(',').str.title()
        df['Commissioner / Provider Code'] = df['Commissioner / Provider Code'].str.slice(0, -2)
        df['sub_location'] = df['Commissioner / Provider Code'].map(mapping)

    # Standardise date columns
    df.rename(columns={'Year Month': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m')
    df['formatted_date'] = df['date'].dt.strftime('%b %Y')

    # Always calculate monthly period for trend logic
    df['date_period'] = df['date'].dt.to_period('M').dt.to_timestamp()

    # Add 'period_tag' for trend analysis (recent / previous / other)
    latest_15_months = df['date_period'].drop_duplicates().sort_values(ascending=False).head(15).tolist()

    # Define tagged time periods
    recent_3m = latest_15_months[:3]
    prev_3m = latest_15_months[3:6]
    last_year_3m = latest_15_months[12:15]

    def label_period(period):
        if period in recent_3m:
            return "recent"
        elif period in prev_3m:
            return "previous"
        elif period in last_year_3m:
            return "last_year"
        else:
            return "other"

    df['period_tag'] = df['date_period'].apply(label_period)

    # Ensure all required columns exist
    for col in ['ADQ Usage', 'DDD Usage']:
        if col not in df.columns:
            df[col] = 0

    return df


# Aggregate prescribing data to Chemical level based on a certain index
def aggregate_substance_data(df, group_cols):
    summary = df.groupby(group_cols, as_index=False)[['Items', 'Actual Cost', 'ADQ Usage', 'DDD Usage']].sum().round(1)
    base = df.drop_duplicates(subset=group_cols).drop(columns=['Items', 'Actual Cost', 'ADQ Usage', 'DDD Usage', 'BNF Chemical Substance plus Code'])
    merged = pd.merge(base, summary, on=group_cols)
    merged['BNF Chemical Substance plus Code'] = 'Drugs Aggregated'
    return merged

# Load data based on selected dataset type
@st.cache_data
def load_data(dataset_type):
    icb_path = f"__{dataset_type} - ICB Dashboard.csv"
    icb_data_raw = pd.read_csv(icb_path)

    if dataset_type == "High Cost Drugs":
        icb_data_preprocessed = icb_data_raw  # Already preprocessed elsewhere
        national_data_preprocessed = None     # No national data
    else:
        icb_data_preprocessed = preprocess_prescribing_data(
            icb_data_raw, is_national=False, mapping=sicbl_legend_mapping
        )

        national_path = f"__{dataset_type} - ICB Dashboard NATIONAL.csv"
        national_data_raw = pd.read_csv(national_path)
        national_data_preprocessed = preprocess_prescribing_data(
            national_data_raw, is_national=True
        )

    return icb_data_preprocessed, national_data_preprocessed


# Load COPD list size data
copd_list_size_df = pd.read_csv("copd_register_may25.csv") # this file 
copd_list_size_df['Practice'] = copd_list_size_df['Practice'].str.title()
copd_list_size_df = copd_list_size_df[['Practice', 'COPD Register']]
copd_list_size_df.rename(columns={'COPD Register': 'COPD List Size'}, inplace=True)


## ── UI: Header and Select Boxes ────────────────
col1, col2 = st.columns([3, 1]) # two column layout

with col1:
    st.markdown("""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            margin-bottom: 1rem;
        ">
            <div style="
                font-size: clamp(20px, 4vw, 36px);
                font-weight: bold;
            ">
                NENC Medicines Optimisation Workstream Dashboard
            </div>
            <div style="
                background-color: #ffc107;
                color: black;
                font-size: 0.9em;
                font-weight: bold;
                padding: 2px 8px;
                border-radius: 4px;
                margin-top: 0.3em;
            ">
                BETA
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    dataset_type = st.selectbox(
        "Select Dataset:",
        options=list(dataset_measures.keys()),
        index=list(dataset_measures.keys()).index(initial_dataset)
    )

    # Assign measure options from central map
    measure_options = dataset_measures.get(dataset_type, [])

    # If the initial_measure is valid for this dataset, keep it. Otherwise, reset to first option.
    if initial_measure in measure_options:
        default_measure_index = measure_options.index(initial_measure)
        resolved_measure = initial_measure
    else:
        default_measure_index = 0
        resolved_measure = measure_options[0]  # fallback to first valid measure

    measure_type = st.selectbox(
        "Select Measure:",
        options=measure_options,
        index=default_measure_index
    )

    # Show cost-per-item slider if High Cost Drugs is selected
    if dataset_type == "High Cost Drugs":
        cost_filter_threshold = st.slider(
            "Minimum Cost per Item to include:",
            min_value=25.0,
            max_value=3000.0,
            value=25.0,
            step=25.0,
            format="£%0.0f"
        )
    else:
        cost_filter_threshold = None


## ── Data Loading ───────────────────────────────
icb_data_preprocessed, national_data_preprocessed = load_data(dataset_type)

# ── Filter High Cost Drugs to those used in <50% of practices ──────────────
if dataset_type == "High Cost Drugs":
    all_hcd_data = icb_data_preprocessed.copy()

    # Total number of unique practices
    total_practices = all_hcd_data["Practice"].nunique()
    threshold = total_practices * 0.5

    # Count how many practices prescribe each drug
    hcd_practice_counts = (
        all_hcd_data.groupby("BNF Presentation plus Code")["Practice"]
        .nunique()
        .reset_index()
        .rename(columns={"Practice": "num_practices"})
    )

    # Filter to only those under threshold
    low_distribution_drugs = hcd_practice_counts[
        hcd_practice_counts["num_practices"] < threshold
    ]["BNF Presentation plus Code"]

    # Apply this filter globally
    icb_data_preprocessed = icb_data_preprocessed[
        icb_data_preprocessed["BNF Presentation plus Code"].isin(low_distribution_drugs)
    ]

    # Apply cost filter if slider threshold is set
    if cost_filter_threshold is not None:
        icb_data_preprocessed = icb_data_preprocessed[
            icb_data_preprocessed["3m Average Cost per Item"] >= cost_filter_threshold
        ]


# Decide which numerator and denominator column to use
numerator_column = measure_metadata[measure_type]["numerator_column"]
denominator_column = measure_metadata[measure_type]["denominator_column"]

# Merge and filter COPD List Size if needed
if dataset_type != "High Cost Drugs":
    icb_data_preprocessed = icb_data_preprocessed.merge(
        copd_list_size_df,
        on='Practice',
        how='left'
    )

    if measure_metadata[measure_type]["denominator_column"] == "COPD List Size":  # Exclude rows with missing COPD List Size if measure requires it
        icb_data_preprocessed = icb_data_preprocessed[icb_data_preprocessed["COPD List Size"].notna()]


## ── Aggregation and Measure Calculation ─────────

# Apply aggregation function
icb_data_aggregated = aggregate_substance_data(icb_data_preprocessed, ['Practice', 'date_period']) # indexed by both practice and date_period. date period helps standardise grouping for bar chart.

# Reattach 'period_tag' from the original preprocessed data
if 'period_tag' in icb_data_preprocessed.columns:
    period_tags = icb_data_preprocessed[['Practice', 'date_period', 'period_tag']].drop_duplicates()

    # Check matches
    matches = icb_data_aggregated.merge(
        period_tags,
        on=['Practice', 'date_period'],
        how='inner'
    )
    # Merge safely without duplicate columns
    icb_data_aggregated = icb_data_aggregated.merge(
        period_tags,
        on=['Practice', 'date_period'],
        how='left',
        suffixes=('', '_drop')
    )
    icb_data_aggregated.drop(columns=[col for col in icb_data_aggregated.columns if col.endswith('_drop')], inplace=True)

# Only apply aggregation function to national data where it exists
if national_data_preprocessed is not None:
    national_data_raw_merged = aggregate_substance_data(national_data_preprocessed, ['date'])  # indexed by date, which is more suitable for line chart

    # Calculate national monthly rates for line chart
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

# Filter to latest 3 months
recent_data = icb_data_aggregated[icb_data_aggregated['period_tag'] == 'recent']

# STEP 1: Sum numerators over latest 3 months
if dataset_type == "High Cost Drugs":
    sum_numerators = recent_data.groupby('Practice', as_index=False)[['Actual Cost', 'Items']].sum().round(1)
else:
    sum_numerators = recent_data.groupby('Practice', as_index=False)[
        ['Actual Cost', 'Items', 'ADQ Usage', 'DDD Usage']
    ].sum().round(1)

# STEP 2: Mean denominator (List Size) over latest 3 months
if dataset_type == "High Cost Drugs":
    mean_denominators = recent_data.groupby('Practice', as_index=False)[['List Size']].mean().round(1)
else:
    mean_denominators = recent_data.groupby('Practice', as_index=False)[['List Size', 'COPD List Size']].mean().round(1)

# STEP 3: Merge together
means = pd.merge(sum_numerators, mean_denominators, on='Practice')

# Optional: drop unnecessary columns from base
columns_to_drop = [col for col in ['Actual Cost', 'Items', 'ADQ Usage', 'DDD Usage', 'List Size', 'COPD List Size', 'date', 'formatted_date'] if col in icb_data_aggregated.columns]
base_means = icb_data_aggregated.drop_duplicates(subset='Practice').drop(columns=columns_to_drop)

# STEP 4: Merge base info with new means
icb_means_merged = pd.merge(base_means, means, on='Practice')

# STEP 5: Calculate rate
icb_means_merged[measure_type] = (
    (icb_means_merged[numerator_column] / icb_means_merged[denominator_column]) * 1000
).round(1)

# Calculate PREVIOUS rates using 4-6m data (only for non-High Cost Drugs)
if dataset_type != "High Cost Drugs":
    previous_data = icb_data_aggregated[icb_data_aggregated['period_tag'] == 'previous']

    # STEP 1: Sum numerators over previous 3 months
    prev_sum_numerators = previous_data.groupby('Practice', as_index=False)[
        ['Actual Cost', 'Items', 'ADQ Usage', 'DDD Usage']
    ].sum().round(1)

    # STEP 2: Mean denominators over previous 3 months
    prev_mean_denominators = previous_data.groupby('Practice', as_index=False)[
        ['List Size', 'COPD List Size']
    ].mean().round(1)

    # STEP 3: Merge numerator and denominator
    previous_means = pd.merge(prev_sum_numerators, prev_mean_denominators, on='Practice')

    # STEP 4: Merge with base (which already contains recent values), creating new cols with _prev suffixed
    icb_means_merged = pd.merge(
        icb_means_merged, previous_means, on='Practice', suffixes=('', '_prev')
    )

    # STEP 5: Calculate previous-period rate using same formula
    icb_means_merged[f"{measure_type} (previous)"] = (
        icb_means_merged[f"{numerator_column}_prev"] / icb_means_merged[f"{denominator_column}_prev"] * 1000
    ).round(1)

    # STEP 6: Calculate % CHANGE
    icb_means_merged["Δ from previous"] = (
        (icb_means_merged[measure_type] - icb_means_merged[f"{measure_type} (previous)"]) /
        icb_means_merged[f"{measure_type} (previous)"] * 100
    ).round(1)

# Calculate LAST YEAR rates using matching period 12 months ago
if dataset_type != "High Cost Drugs":
    last_year_data = icb_data_aggregated[icb_data_aggregated['period_tag'] == 'last_year']

    # STEP 1: Sum numerators over last-year 3 months
    last_year_sum_numerators = last_year_data.groupby('Practice', as_index=False)[
        ['Actual Cost', 'Items', 'ADQ Usage', 'DDD Usage']
    ].sum().round(1)

    # STEP 2: Mean denominators over last-year 3 months
    last_year_mean_denominators = last_year_data.groupby('Practice', as_index=False)[
        ['List Size', 'COPD List Size']
    ].mean().round(1)

    # STEP 3: Merge numerator and denominator
    last_year_means = pd.merge(last_year_sum_numerators, last_year_mean_denominators, on='Practice')

    # STEP 4: Merge into main df, add suffix
    icb_means_merged = pd.merge(
        icb_means_merged, last_year_means, on='Practice', suffixes=('', '_lastyear')
    )

    # STEP 5: Calculate last-year-period rate
    icb_means_merged[f"{measure_type} (last year)"] = (
        icb_means_merged[f"{numerator_column}_lastyear"] /
        icb_means_merged[f"{denominator_column}_lastyear"] * 1000
    ).round(1)

    # STEP 6: Calculate % CHANGE FROM LAST YEAR
    icb_means_merged["Δ from last year"] = (
        (icb_means_merged[measure_type] - icb_means_merged[f"{measure_type} (last year)"]) /
        icb_means_merged[f"{measure_type} (last year)"] * 100
    ).round(1)


# Round item count
icb_means_merged['Items'] = icb_means_merged['Items'].round(0).astype(int) # Round item count
icb_means_merged.rename(columns={'Items': 'Items (monthly average)'}, inplace=True) # Rename items as monthly average for clarity

# Calculate ICB Average Values
total_numerator = sum_numerators[numerator_column].sum()
total_denominator = mean_denominators[denominator_column].sum()
icb_average_value = (total_numerator / total_denominator) * 1000


# ── Line chart: Calculate Monthly Rate Columns ──────────

# Calculate monthly rates for ICB data
icb_data_aggregated[measure_type] = (
    (icb_data_aggregated[numerator_column] / icb_data_aggregated[denominator_column]) * 1000
).round(1)



### STREAMLIT LAYOUT ------------

# ── Bar Chart Section ─────────────────────────────

st.header(f'Practice comparisons: {dataset_type} - {measure_type}')

# Two-column layout for sublocation and practice selector
col1, col2 = st.columns([2, 2])

# ── Sub-location Dropdown (col1)
with col1:
    subloc_options = ["Show all"] + list(sub_location_colors.keys())
    selected_sublocation = st.selectbox(
        "Select Sub-location:",
        options=subloc_options,
        index=subloc_options.index(initial_subloc) if initial_subloc in subloc_options else 0,
        key="subloc_selector"
    )

# Convert selection to a list for filtering
if selected_sublocation == "Show all":
    selected_sublocations = list(sub_location_colors.keys())
else:
    selected_sublocations = [selected_sublocation]


# ── Toggle for bar chart y-axis mode ─────────────
use_delta_chart = False
use_year_change = False
if dataset_type != "High Cost Drugs":
    mode_option = st.radio(
        "",
        ["Current Rate (latest 3m)", "Change from Previous 3m", "Change from Same 3m Last Year"],
        index=0,
        horizontal=True
    )
    use_delta_chart = mode_option == "Change from Previous 3m"
    use_year_change = mode_option == "Change from Same 3m Last Year"

# Filter by selected sublocations for plotting
filtered_data = icb_means_merged[icb_means_merged['sub_location'].isin(selected_sublocations)]

# ── Conditional Practice Dropdown (col2)
selected_practice = None
with col2:
    if selected_sublocation != "Show all":
        practices = filtered_data['Practice'].sort_values().unique()
        practice_options = ["None"] + list(practices)
        default_index = practice_options.index(initial_highlight) if initial_highlight in practice_options else 0

        selected_practice_option = st.selectbox(
            "Select Practice:",
            options=practice_options,
            index=default_index
        )

        if selected_practice_option != "None":
            selected_practice = selected_practice_option


# ── Legend (only when multiple sublocations)
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

# Determine which column to use for y-axis
if use_delta_chart:
    y_axis_column = "Δ from previous"
    y_axis_label = f"{dataset_type} {measure_type} Δ from previous 3m"
elif use_year_change:
    y_axis_column = "Δ from last year"
    y_axis_label = f"{dataset_type} {measure_type} Δ from same 3m last year"
else:
    y_axis_column = measure_type
    y_axis_label = f"{dataset_type} {measure_type}"

delta_column = None
if dataset_type != "High Cost Drugs":
    if use_delta_chart:
        delta_column = "Δ from previous"
    elif use_year_change:
        delta_column = "Δ from last year"


# Render Bar Chart
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
            "Select view",
            options=["View as scatterplot", "View as table"],
            horizontal=True,
            label_visibility="collapsed"
        )

        practice_data = icb_data_preprocessed[
            icb_data_preprocessed["Practice"] == selected_practice
        ].copy()

        # Get top 100 BNF presentation names by spend
        top_presentations = (
            practice_data.groupby("BNF Presentation plus Code", as_index=False)[["Items", "Actual Cost"]]
            .sum()
            .sort_values("Actual Cost", ascending=False)
            .head(100)["BNF Presentation plus Code"]
            .sort_values()
            .tolist()
        )

        if view_mode == "View as scatterplot":
            top100_summary = (
                practice_data[practice_data["BNF Presentation plus Code"].isin(top_presentations)]
                .groupby("BNF Presentation plus Code", as_index=False)[["Items", "Actual Cost"]]
                .sum()
                .sort_values("Actual Cost", ascending=False)
            )
            scatter_fig = plot_high_cost_drugs_scatter(top100_summary)
            if scatter_fig:
                st.plotly_chart(scatter_fig, use_container_width=True)
            else:
                st.info("No data available for the selected practice.")

        elif view_mode == "View as table":
            top_drugs_grouped = (
                practice_data[practice_data["BNF Presentation plus Code"].isin(top_presentations)]
                .groupby("BNF Presentation plus Code", as_index=False)[["Actual Cost", "Items"]]
                .sum()
            )

            top_drugs_grouped["Average Cost per Item"] = (
                top_drugs_grouped["Actual Cost"] / top_drugs_grouped["Items"]
            )

            top_drugs_table = top_drugs_grouped.sort_values("Actual Cost", ascending=False)


            styled_table = top_drugs_table.style \
                .format({
                    "Actual Cost": "£{:,.0f}",
                    "Items": "{:,.0f}",
                    "Average Cost per Item": "£{:,.0f}"
                }) \
                .background_gradient(subset=["Average Cost per Item"], cmap="Reds")

            st.dataframe(styled_table, use_container_width=True, hide_index=True)

    elif selected_sublocation and selected_sublocation != "Show all":
        st.header(f"Top 100 High Cost Drugs (latest 3m) in {selected_sublocation}")

        view_mode = st.radio(
            "Select view",
            options=["View as scatterplot", "View as table"],
            horizontal=True,
            label_visibility="collapsed"
        )

        subloc_data = icb_data_preprocessed[
            icb_data_preprocessed["sub_location"].str.strip().str.casefold() == selected_sublocation.strip().casefold()
        ].copy()

        # Get top 100 BNF presentation names
        top_presentations = (
            subloc_data.groupby("BNF Presentation plus Code", as_index=False)[["Items", "Actual Cost"]]
            .sum()
            .sort_values("Actual Cost", ascending=False)
            .head(100)["BNF Presentation plus Code"]
            .sort_values()
            .tolist()
        )

        if view_mode == "View as scatterplot":
            top100_summary = (
                subloc_data[subloc_data["BNF Presentation plus Code"].isin(top_presentations)]
                .groupby("BNF Presentation plus Code", as_index=False)[["Items", "Actual Cost"]]
                .sum()
                .sort_values("Actual Cost", ascending=False)
            )
            scatter_fig = plot_high_cost_drugs_scatter(top100_summary)
            if scatter_fig:
                st.plotly_chart(scatter_fig, use_container_width=True)
            else:
                st.info("No data available for the selected sublocation.")

        elif view_mode == "View as table":
            top_drugs_grouped = (
                subloc_data[subloc_data["BNF Presentation plus Code"].isin(top_presentations)]
                .groupby("BNF Presentation plus Code", as_index=False)[["Actual Cost", "Items"]]
                .sum()
            )

            top_drugs_grouped["Average Cost per Item"] = (
                top_drugs_grouped["Actual Cost"] / top_drugs_grouped["Items"]
            )

            top_drugs_table = top_drugs_grouped.sort_values("Actual Cost", ascending=False)

            styled_table = top_drugs_table.style \
                .format({
                    "Actual Cost": "£{:,.0f}",
                    "Items": "{:,.0f}",
                    "Average Cost per Item": "£{:,.0f}"
                }) \
                .background_gradient(subset=["Average Cost per Item"], cmap="Reds")

            st.dataframe(styled_table, use_container_width=True, hide_index=True)

        selected_presentation = st.selectbox("Select a drug to view practice-level breakdown:", top_presentations)

        if selected_presentation:
            subset = subloc_data[subloc_data["BNF Presentation plus Code"] == selected_presentation]
            num_practices = subset["Practice"].nunique()
            st.subheader(f"{num_practices} practices prescribe {selected_presentation}")

            practice_breakdown = (
                subset.groupby("Practice", as_index=False)[["Actual Cost", "Items"]]
                .sum()
                .sort_values("Actual Cost", ascending=False)
            )

            styled_practice_breakdown = practice_breakdown.style \
                .format({
                    "Actual Cost": "£{:,.0f}",
                    "Items": "{:,.0f}",
                })

            st.dataframe(styled_practice_breakdown, use_container_width=True, hide_index=True)


else:

    # Existing line chart section for other datasets

    st.header(f'Trend analysis: {dataset_type} - {measure_type}')

    # ── Redisplay Legend Above Line Chart ─────────────────────
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
            icb_data_aggregated,
            national_data_raw_merged,
            sub_location=selected_sublocation,
            selected_sublocation=selected_sublocation,
            selected_practice=selected_practice,
            measure_type=measure_type,
            dataset_type=dataset_type,
            mode="practice"
        )

        st.plotly_chart(line_fig, use_container_width=True)

    else:
        line_fig = plot_line_chart(
            icb_data_aggregated,
            national_data_raw_merged,
            sub_location=None,
            selected_sublocation=selected_sublocation,
            selected_practice=None,
            measure_type=measure_type,
            dataset_type=dataset_type,
            mode="sublocations",
            sub_location_colors=sub_location_colors
        )

        st.plotly_chart(line_fig, use_container_width=True)


    # Deeplinking param updates
    if ENABLE_DEEPLINKING:
        current_params = {
            "dataset": initial_dataset,
            "measure": initial_measure,
            "sublocation": initial_subloc,
            "highlight": initial_highlight,
        }

        new_params = {
            "dataset": dataset_type,
            "measure": measure_type,
            "sublocation": selected_sublocation,
            "highlight": selected_practice or "None"
        }

        if current_params != new_params and not st.session_state.query_updated:
            st.query_params.update(new_params)
            st.session_state.query_updated = True