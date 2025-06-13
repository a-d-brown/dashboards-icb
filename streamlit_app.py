## ── Imports and Config ────────────────────────
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from chart_utils import plot_icb_bar_chart, plot_line_chart

pio.templates.default = 'simple_white'
st.set_page_config(page_title="ICB Workstream Dashboard", layout="wide")

## ── CSS Styling ────────────────────────────────
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

# Define available measures per dataset
dataset_measures = {
    "SABAs": ["Spend per 1000 Patients", "Items per 1000 Patients"],
    "Opioids": ["Spend per 1000 Patients", "Items per 1000 Patients", "ADQ per 1000 Patients"],
    "Lidocaine Patches": ["Spend per 1000 Patients", "Items per 1000 Patients"],
    "Antibacterials": ["Spend per 1000 Patients", "Items per 1000 Patients", "DDD per 1000 Patients"],
    "Closed Triple Inhalers": ["Spend per 1000 COPD Patients", "Items per 1000 COPD Patients"]
}

# Define which measure columns should be referred to when each measure type if selected
measure_column_lookup = {
    "Spend per 1000 Patients": "Spend per 1000 Patients",
    "Spend per 1000 COPD Patients": "Spend per 1000 Patients",
    "Items per 1000 Patients": "Items per 1000 Patients",
    "Items per 1000 COPD Patients": "Items per 1000 Patients",
    "ADQ per 1000 Patients": "ADQ per 1000 Patients",
    "DDD per 1000 Patients": "DDD per 1000 Patients"
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
    file_map = {
        'SABAs': ("__SABAs - ICB Dashboard.csv", "__SABAs - ICB Dashboard NATIONAL.csv"),
        'Opioids': ("__Opioids - ICB Dashboard.csv", "__Opioids - ICB Dashboard NATIONAL.csv"),
        'Lidocaine Patches': ("__Lidocaine - ICB Dashboard.csv", "__Lidocaine - ICB Dashboard NATIONAL.csv"),
        'Antibacterials': ("__Antibacterials - ICB Dashboard.csv", "__Antibacterials - ICB Dashboard NATIONAL.csv"),
        'Closed Triple Inhalers': ("__Triple Inhalers - ICB Dashboard.csv", "__Triple Inhalers - ICB Dashboard NATIONAL.csv")

    }

    icb_path, national_path = file_map[dataset_type]
    icb_data_raw = pd.read_csv(icb_path)
    national_data_raw = pd.read_csv(national_path)

    icb_data_preprocessed = preprocess_prescribing_data(icb_data_raw, is_national=False, mapping=sicbl_legend_mapping)
    national_data_preprocessed = preprocess_prescribing_data(national_data_raw, is_national=True)

    return icb_data_preprocessed, national_data_preprocessed

# Load COPD list size data
copd_list_size_df = pd.read_csv("copd_patients_23_24.csv")
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
        key="data_type_selector"
    )

    # Assign measure options from central map
    measure_options = dataset_measures.get(dataset_type, [])
    measure_type = st.selectbox(
        "Select Measure:",
        options=measure_options,
        key="measure_selector"
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
if measure_type in ["Spend per 1000 COPD Patients", "Items per 1000 COPD Patients"]:
    icb_data_preprocessed = icb_data_preprocessed[icb_data_preprocessed['COPD List Size'].notna()]



## ── Aggregation and Measure Calculation ─────────


# Apply aggregation function
icb_data_raw_merged = aggregate_substance_data(icb_data_preprocessed, ['Practice', 'date_period']) # indexed by both practice and date_period. date period helps standardise grouping for bar chart.
national_data_raw_merged = aggregate_substance_data(national_data_preprocessed, ['date']) # only date index needed as one-dimensional data. plotly expects datetime[ns64] for line chart.


# Calculate mean spend and items in last 3m
recent_data = icb_data_raw_merged.sort_values('date', ascending=False).groupby('Practice').head(3) # Get latest 3m into a df
means = recent_data.groupby('Practice', as_index=False)[['List Size', 'COPD List Size', 'Actual Cost', 'Items', 'ADQ Usage', 'DDD Usage']].mean().round(1) # Calculate means
base_means = icb_data_raw_merged.drop_duplicates(subset='Practice').drop(columns=['List Size', 'COPD List Size','Actual Cost', 'Items', 'ADQ Usage', 'DDD Usage', 'date', 'formatted_date']) # Create metadata base
icb_means_merged = pd.merge(base_means, means, on='Practice') # Merge base with means

# Decide which denominator column to use
if measure_type in ["Spend per 1000 COPD Patients", "Items per 1000 COPD Patients"]:
    denominator_column = "COPD List Size"
else:
    denominator_column = "List Size"

# Calculate 3m mean rates
icb_means_merged['Spend per 1000 Patients'] = (         # Calculate Spend per 1000 Patients
    (icb_means_merged['Actual Cost'] / icb_means_merged[denominator_column]) * 1000
).round(1)
icb_means_merged['Items per 1000 Patients'] = (         # Calculate Items per 1000 Patients
    (icb_means_merged['Items'] / icb_means_merged[denominator_column]) * 1000
).round(1)
icb_means_merged['ADQ per 1000 Patients'] = (         # Calculate Items per 1000 Patients
    (icb_means_merged['ADQ Usage'] / icb_means_merged[denominator_column]) * 1000
).round(1)
icb_means_merged['DDD per 1000 Patients'] = (         # Calculate Items per 1000 Patients
    (icb_means_merged['DDD Usage'] / icb_means_merged[denominator_column]) * 1000
).round(1)

# Round item count
icb_means_merged['Items'] = icb_means_merged['Items'].round(0).astype(int) # Round item count
icb_means_merged.rename(columns={'Items': 'Items (monthly average)'}, inplace=True) # Rename items as monthly average for clarity

# Calculate all possible ICB Average Values
total_actual_cost = recent_data['Actual Cost'].sum()
total_items = recent_data['Items'].sum()
total_adq = recent_data['ADQ Usage'].sum()
total_ddd = recent_data['DDD Usage'].sum()
total_list_size = recent_data[denominator_column].sum()

icb_average_spend = (total_actual_cost / total_list_size) * 1000  # Spend per 1000 Patients
icb_average_items = (total_items / total_list_size) * 1000  # Items per 1000 Patients
icb_average_adq = (total_adq / total_list_size) * 1000  # ADQ per 1000 Patients
icb_average_ddd = (total_ddd / total_list_size) * 1000  # DDD per 1000 Patients


# Set ICB average value ot show based on 'Select Measure' value
if measure_type == "Spend per 1000 Patients":
    icb_average_value = icb_average_spend
elif measure_type == "Items per 1000 Patients":
    icb_average_value = icb_average_items
elif measure_type == "ADQ per 1000 Patients":
    icb_average_value = icb_average_adq
elif measure_type == "DDD per 1000 Patients":
    icb_average_value = icb_average_ddd
elif measure_type == "Spend per 1000 COPD Patients":
    icb_average_value = icb_average_spend
elif measure_type == "Items per 1000 COPD Patients":
    icb_average_value = icb_average_items


# ── Line chart: Calculate Monthly Rate Columns ──────────

# Calculate monthly rates for ICB data
icb_data_raw_merged['Spend per 1000 Patients'] = ((icb_data_raw_merged['Actual Cost'] / icb_data_raw_merged[denominator_column]) * 1000).round(1)
icb_data_raw_merged['Items per 1000 Patients'] = ((icb_data_raw_merged['Items'] / icb_data_raw_merged[denominator_column]) * 1000).round(1)
icb_data_raw_merged['ADQ per 1000 Patients'] = ((icb_data_raw_merged['ADQ Usage'] / icb_data_raw_merged[denominator_column]) * 1000).round(1)
icb_data_raw_merged['DDD per 1000 Patients'] = ((icb_data_raw_merged['DDD Usage'] / icb_data_raw_merged[denominator_column]) * 1000).round(1)

# Calculate monthly rates for national data
if measure_type in ["Spend per 1000 COPD Patients"]:
    national_data_raw_merged['Spend per 1000 Patients'] = (
        (national_data_raw_merged['Actual Cost'] / NATIONAL_COPD_LIST_SIZE) * 1000
    ).round(1)
else:
    national_data_raw_merged['Spend per 1000 Patients'] = (
        (national_data_raw_merged['Actual Cost'] / national_data_raw_merged['List Size']) * 1000
    ).round(1)

if measure_type == "Items per 1000 COPD Patients":
    national_data_raw_merged['Items per 1000 Patients'] = (
        (national_data_raw_merged['Items'] / NATIONAL_COPD_LIST_SIZE) * 1000
    ).round(1)
else:
    national_data_raw_merged['Items per 1000 Patients'] = (
        (national_data_raw_merged['Items'] / national_data_raw_merged['List Size']) * 1000
    ).round(1)

national_data_raw_merged['ADQ per 1000 Patients'] = ((national_data_raw_merged['ADQ Usage'] / national_data_raw_merged['List Size']) * 1000).round(1)
national_data_raw_merged['DDD per 1000 Patients'] = ((national_data_raw_merged['DDD Usage'] / national_data_raw_merged['List Size']) * 1000).round(1)


### STREAMLIT LAYOUT ------------

# ── Bar Chart Section ─────────────────────────────

st.header(f'{measure_type} on {dataset_type}: ICB-wide comparison in the last 3m')

# Two-column layout for sublocation and practice selector
col1, col2 = st.columns([2, 2])

# ── Sub-location Dropdown (col1)
with col1:
    subloc_options = ["Show all"] + list(sub_location_colors.keys())
    selected_subloc_option = st.selectbox(
        "Select Sub-location:",
        options=subloc_options
    )

# Convert selection to a list for filtering
if selected_subloc_option == "Show all":
    selected_sublocations = list(sub_location_colors.keys())
else:
    selected_sublocations = [selected_subloc_option]

# Filter data
filtered_data = icb_means_merged[icb_means_merged['sub_location'].isin(selected_sublocations)]

# ── Conditional Practice Dropdown (col2)
highlighted_practice = None
with col2:
    if len(selected_sublocations) == 1:
        practices = filtered_data['Practice'].sort_values().unique()
        selected_practice_option = st.selectbox("Highlight a practice (optional):", ["None"] + list(practices))
        if selected_practice_option != "None":
            highlighted_practice = selected_practice_option

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

# Determine measure column based on selected measure type
measure_column = measure_column_lookup.get(measure_type)

# Render Bar Chart
st.subheader("🔍 Preview of Bar Chart Data")
st.dataframe(filtered_data[['Practice', 'sub_location', measure_column, 'Items (monthly average)']].sort_values(by=measure_column, ascending=False).head(10))

bar_fig = plot_icb_bar_chart(
    filtered_data=filtered_data,
    measure_type=measure_type,
    measure_column=measure_column,
    sub_location_colors=sub_location_colors,
    icb_average_value=icb_average_value,
    dataset_type=dataset_type,
    highlighted_practice=highlighted_practice
)

st.plotly_chart(bar_fig, use_container_width=True)

st.markdown("---")



# ── Line Chart Section ─────────────────────────

st.header(f'{measure_type} on {dataset_type}: Local Trends')

col1, col2 = st.columns(2)

with col1:
    sicbl_options = sorted(icb_data_raw_merged['sub_location'].dropna().unique())
    selected_sublocation = st.selectbox("Select Sub-location:", options=sicbl_options)

with col2:
    filtered_practices = icb_data_raw_merged[icb_data_raw_merged['sub_location'] == selected_sublocation]
    practice_options = sorted(filtered_practices['Practice'].unique())
    selected_practice = st.selectbox("Select Practice:", options=practice_options)

# Render Line Chart

line_fig = plot_line_chart(
    icb_data_raw_merged,
    national_data_raw_merged,
    selected_sublocation,
    selected_practice,
    measure_type,
    measure_column,
    dataset_type
)

st.plotly_chart(line_fig, use_container_width=True)
