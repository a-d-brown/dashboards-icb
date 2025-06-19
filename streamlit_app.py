## ── Imports and Config ────────────────────────
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from chart_utils import plot_icb_bar_chart, plot_line_chart

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
    "Bath & Shower Emollients": ["Spend per 1000 COPD Patients", "Items per 1000 COPD Patients"]
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
    }
}


NATIONAL_COPD_LIST_SIZE = 1175163


## ── Functions: Preprocessing and Loading ───────

# Preprocess data to standardised format
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

    if not is_national:
        df['date_period'] = df['date'].dt.to_period('M')

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
    national_path = f"__{dataset_type} - ICB Dashboard NATIONAL.csv"

    icb_data_raw = pd.read_csv(icb_path)
    national_data_raw = pd.read_csv(national_path)

    icb_data_preprocessed = preprocess_prescribing_data(icb_data_raw, is_national=False, mapping=sicbl_legend_mapping)
    national_data_preprocessed = preprocess_prescribing_data(national_data_raw, is_national=True)

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

## ── Data Loading ───────────────────────────────
icb_data_preprocessed, national_data_preprocessed = load_data(dataset_type)

# Merge COPD List Size into ICB data
icb_data_preprocessed = icb_data_preprocessed.merge(
    copd_list_size_df,
    on='Practice',
    how='left'
)

# Exclude rows with missing COPD List Size if measure requires it
if measure_metadata[measure_type]["denominator_column"] == "COPD List Size":
    icb_data_preprocessed = icb_data_preprocessed[icb_data_preprocessed["COPD List Size"].notna()]


## ── Aggregation and Measure Calculation ─────────

# Apply aggregation function
icb_data_raw_merged = aggregate_substance_data(icb_data_preprocessed, ['Practice', 'date_period']) # indexed by both practice and date_period. date period helps standardise grouping for bar chart.
national_data_raw_merged = aggregate_substance_data(national_data_preprocessed, ['date']) # only date index needed as one-dimensional data. plotly expects datetime[ns64] for line chart.

# Calculate mean spend and items in last 3m
recent_data = icb_data_raw_merged.sort_values('date', ascending=False).groupby('Practice').head(3) # Get latest 3m into a df
means = recent_data.groupby('Practice', as_index=False)[['List Size', 'COPD List Size', 'Actual Cost', 'Items', 'ADQ Usage', 'DDD Usage']].mean().round(1) # Calculate means
base_means = icb_data_raw_merged.drop_duplicates(subset='Practice').drop(columns=['List Size', 'COPD List Size','Actual Cost', 'Items', 'ADQ Usage', 'DDD Usage', 'date', 'formatted_date']) # Create metadata base
icb_means_merged = pd.merge(base_means, means, on='Practice') # Merge base with means

# Decide which numerator and denominator column to use
numerator_column = measure_metadata[measure_type]["numerator_column"]
denominator_column = measure_metadata[measure_type]["denominator_column"]

# Calculate 3m mean rates
icb_means_merged[measure_type] = (
    (icb_means_merged[numerator_column] / icb_means_merged[denominator_column]) * 1000
).round(1)

# Round item count
icb_means_merged['Items'] = icb_means_merged['Items'].round(0).astype(int) # Round item count
icb_means_merged.rename(columns={'Items': 'Items (monthly average)'}, inplace=True) # Rename items as monthly average for clarity

# Calculate ICB Average Values
total_numerator = recent_data[numerator_column].sum()
total_denominator = recent_data[denominator_column].sum()
icb_average_value = (total_numerator / total_denominator) * 1000


# ── Line chart: Calculate Monthly Rate Columns ──────────

# Calculate monthly rates for ICB data
icb_data_raw_merged[measure_type] = (
    (icb_data_raw_merged[numerator_column] / icb_data_raw_merged[denominator_column]) * 1000
).round(1)

# Calculate monthly rates for national data
if denominator_column == "COPD List Size":
    national_data_raw_merged[measure_type] = (
        (national_data_raw_merged[numerator_column] / NATIONAL_COPD_LIST_SIZE) * 1000
    ).round(1)
else:
    national_data_raw_merged[measure_type] = (
        (national_data_raw_merged[numerator_column] / national_data_raw_merged[denominator_column]) * 1000
    ).round(1)



### STREAMLIT LAYOUT ------------

# ── Bar Chart Section ─────────────────────────────

st.header(f'Practice comparisons (latest 3m): {measure_type} on {dataset_type}')

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

# Filter data
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

# Render Bar Chart
bar_fig = plot_icb_bar_chart(
    filtered_data=filtered_data,
    measure_type=measure_type,
    sub_location_colors=sub_location_colors,
    icb_average_value=icb_average_value,
    dataset_type=dataset_type,
    measure_metadata=measure_metadata,
    selected_practice=selected_practice
)

st.plotly_chart(bar_fig, use_container_width=True)

st.markdown("---")


# ── Line Chart Section ─────────────────────────

st.header(f'Trend analysis: {measure_type} on {dataset_type}')

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
        icb_data_raw_merged,
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
        icb_data_raw_merged,
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




