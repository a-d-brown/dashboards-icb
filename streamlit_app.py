# streamlit_app.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Set minimalist style for Plotly charts
pio.templates.default = 'simple_white'

# Streamlit Page Config
st.set_page_config(page_title="ICB Workstream Dashboard", layout="wide")

# Add custom CSS to adjust the font size of dropdowns
st.markdown("""
    <style>
        .stSelectbox div[data-baseweb="select"] {
            font-size: 18px;
            width: 100%;
            max-width: 800px;
        }
    </style>
""", unsafe_allow_html=True)

# Add mapping for local SICBL names
sicbl_legend_mapping = {
    '84H': 'Durham',
    '00P': 'Sunderland',
    '00L': 'Northumberland',
    '01H': 'Cumbria',
    '13T': 'Newcastle-Gateshead',
    '16C': 'Tees Valley',
    '99C': 'North Tyneside',
    '00N': 'South Tyneside'
}

# Define fixed colors for sub_locations
sub_location_colors = {
    'Durham': '#1f77b4',
    'Sunderland': '#d62728',
    'Northumberland': '#2ca02c',
    'Cumbria': '#9467bd',
    'Newcastle-Gateshead': '#ff7f0e',
    'Tees Valley': '#17becf',
    'North Tyneside': '#e377c2',
    'South Tyneside': '#bcbd22'
}

def format_date_column(df):
    df.rename(columns={'Year Month': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m')
    df['formatted_date'] = df['date'].dt.strftime('%b %Y')
    return df

# Define function to load data based on selected dataset type
def load_data(dataset_type):
    if dataset_type == 'SABAs':
        icb_data_raw = pd.read_csv("__SABAs - ICB Dashboard.csv")
        national_data_raw = pd.read_csv("__SABAs - ICB Dashboard NATIONAL.csv")
    elif dataset_type == 'Opioids':
        icb_data_raw = pd.read_csv("__Opioids - ICB Dashboard.csv")
        national_data_raw = pd.read_csv("__Opioids - ICB Dashboard NATIONAL.csv")
    elif dataset_type == 'Lidocaine':
        icb_data_raw = pd.read_csv("__Lidocaine - ICB Dashboard.csv")
        national_data_raw = pd.read_csv("__Lidocaine - ICB Dashboard NATIONAL.csv")

    
    # Data preprocessing for selected dataset
    icb_data_raw = icb_data_raw[icb_data_raw['PCN'] != 'DUMMY']
    icb_data_raw = icb_data_raw[~icb_data_raw['Practice'].str.contains(r'\( ?[CD] ?\d', na=False)]
    icb_data_raw['Practice'] = icb_data_raw['Practice'].str.rstrip(',').str.title()
    icb_data_raw['Commissioner / Provider Code'] = icb_data_raw['Commissioner / Provider Code'].str.slice(0, -2)
    icb_data_raw['sub_location'] = icb_data_raw['Commissioner / Provider Code'].map(sicbl_legend_mapping)
    icb_data_raw = format_date_column(icb_data_raw)
    icb_data_raw['date_period'] = icb_data_raw['date'].dt.to_period('M')

    national_data_raw = format_date_column(national_data_raw)

    return icb_data_raw, national_data_raw

# Streamlit layout for main title and dropdown side by side
col1, col2 = st.columns([3, 1])

with col1:
    st.title("NENC Medicines Optimisation Workstream Dashboard")

with col2:
    dataset_type = st.selectbox("Select Dataset:", options=["SABAs", "Opioids", "Lidocaine"], key="data_type_selector")

# Load data based on the selected dataset
icb_data_raw, national_data_raw = load_data(dataset_type)

# Aggregate data as before for SABA or Opioid data
summary = icb_data_raw.groupby(['Practice', 'date_period'], as_index=False)[['Items', 'Actual Cost', 'ADQ Usage']].sum().round(1)
base_aggregate = icb_data_raw.drop_duplicates(subset=['Practice', 'date_period']).drop(columns=['Items', 'Actual Cost', 'ADQ Usage', 'BNF Chemical Substance plus Code'])
icb_data_raw_merged = pd.merge(base_aggregate, summary, on=['Practice', 'date_period'])
icb_data_raw_merged['BNF Chemical Substance plus Code'] = 'Drugs Aggregated'

# Aggregate NATIONAL data across chemical substances
national_summary = national_data_raw.groupby(['date'], as_index=False)[['Items', 'Actual Cost', 'ADQ Usage']].sum().round(1)
national_base = national_data_raw.drop_duplicates(subset=['date']).drop(columns=['Items', 'Actual Cost', 'ADQ Usage', 'BNF Chemical Substance plus Code'])
national_data_raw_merged = pd.merge(national_base, national_summary, on=['date'])
national_data_raw_merged['BNF Chemical Substance plus Code'] = 'Drugs Aggregated'

# Bar chart ------------

# Calculate mean spend and items in last 3m
recent_data = icb_data_raw_merged.sort_values('date', ascending=False).groupby('Practice').head(3) # Get latest 3m into a df
means = recent_data.groupby('Practice', as_index=False)[['List Size', 'Actual Cost', 'Items', 'ADQ Usage']].mean().round(1) # Calculate means
base_means = icb_data_raw_merged.drop_duplicates(subset='Practice').drop(columns=['List Size', 'Actual Cost', 'Items', 'ADQ Usage', 'date', 'formatted_date']) # Create metadata base
saba_means_merged = pd.merge(base_means, means, on='Practice') # Merge base with means

saba_means_merged['Spend per 1000 Patients'] = (         # Calculate Spend per 1000 Patients
    (saba_means_merged['Actual Cost'] / saba_means_merged['List Size']) * 1000
).round(1)

saba_means_merged['Items'] = saba_means_merged['Items'].round(0).astype(int) # Round item count
saba_means_merged.rename(columns={'Items': 'Items (monthly average)'}, inplace=True) # Rename items as monthly average for clarity

# ICB Average Spend
total_actual_spend = saba_means_merged['Actual Cost'].sum()
total_list_size = saba_means_merged['List Size'].sum()
icb_average_spend = (total_actual_spend / total_list_size) * 1000

# Line Chart ------------

# Calculate items per patient
icb_data_raw_merged['Spend per 1000 Patients'] = ((icb_data_raw_merged['Actual Cost'] / icb_data_raw_merged['List Size']) * 1000).round(1)
national_data_raw_merged['Spend per 1000 Patients'] = ((national_data_raw_merged['Actual Cost'] / national_data_raw_merged['List Size']) * 1000).round(1)

### STREAMLIT LAYOUT ------------

# Bar chart ------------

st.header(f'Spend on {dataset_type}: ICB-wide comparison in the last 3m')

# Initialize session state for toggles
for subloc in sub_location_colors.keys():
    if f"subloc_toggle_{subloc}" not in st.session_state:
        st.session_state[f"subloc_toggle_{subloc}"] = True

cols = st.columns(len(sub_location_colors))
rerun_needed = False

for i, (subloc, color) in enumerate(sub_location_colors.items()):
    with cols[i]:
        row_1 = st.columns([0.15, 0.5])
        is_toggled_off = not st.session_state.get(f"subloc_toggle_{subloc}", False)
        strikethrough_style = "text-decoration: line-through; color: #d3d3d3;" if is_toggled_off else ""

        row_1[0].markdown(f"<div style='width: 16px; height: 16px; background-color: {color}; border-radius: 3px; margin-right: 0px; border: 1px solid #00000033;'></div>", unsafe_allow_html=True)
        row_1[1].markdown(f"<div title='{subloc}' style='font-weight: bold; font-size: 13px; cursor: default; vertical-align: top; text-align: left; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 12vw; min-width: 60px; {strikethrough_style}'>{subloc}</div>", unsafe_allow_html=True)

        row_2 = st.columns([1])
        if row_2[0].button("Toggle", key=f"toggle_{subloc}"):
            st.session_state[f"subloc_toggle_{subloc}"] = not st.session_state.get(f"subloc_toggle_{subloc}", False)
            rerun_needed = True

if rerun_needed:
    st.rerun()

# Collect selected sub_locations
selected_sublocations = [subloc for subloc in sub_location_colors.keys() if st.session_state[f"subloc_toggle_{subloc}"]]

# Filter data based on selected sub_locations
filtered_data = saba_means_merged[saba_means_merged['sub_location'].isin(selected_sublocations)]
show_xticks = len(selected_sublocations) == 1

# Create updated bar chart
bar_dynamic = px.bar(
    filtered_data.sort_values('Spend per 1000 Patients', ascending=False),
    x='Practice',
    y='Spend per 1000 Patients',
    hover_name='Practice',
    hover_data={'sub_location': False, 'Practice': False, 'Spend per 1000 Patients': True, 'Items (monthly average)': True},
    color='sub_location',
    color_discrete_map=sub_location_colors,
)

bar_dynamic.update_layout(
    height=700,
    width=1150,
    xaxis_title='',
    yaxis_title=f'{dataset_type} Spend per 1000 Patients',
    yaxis_tickprefix="£",
    legend_title_text=None,
    xaxis=dict(showticklabels=show_xticks, showgrid=False, tickangle=45, tickfont=dict(size=10)),
    yaxis=dict(showgrid=False),
    legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5),
    showlegend=False,
    margin=dict(r=100)
)

bar_dynamic.update_traces(width=0.6)

bar_dynamic.add_shape(
    type="line", x0=0, x1=1, y0=icb_average_spend, y1=icb_average_spend,
    line=dict(color="black", width=1.5, dash="dash"), xref="paper", yref="y"
)

bar_dynamic.add_annotation(
    x=1, y=icb_average_spend,
    text="ICB Average",
    showarrow=False,
    font=dict(size=12, color="black"),
    xref="paper", yref="y",
    xanchor="left", yanchor="middle"
)

st.plotly_chart(bar_dynamic, use_container_width=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Line Chart ------------

st.header(f"Spend on {dataset_type}: Local Trends")

col1, col2 = st.columns(2)

with col1:
    sicbl_options = sorted(icb_data_raw_merged['sub_location'].dropna().unique())
    selected_sublocation = st.selectbox("Select Sub-location:", options=sicbl_options)

with col2:
    filtered_practices = icb_data_raw_merged[icb_data_raw_merged['sub_location'] == selected_sublocation]
    practice_options = sorted(filtered_practices['Practice'].unique())
    selected_practice = st.selectbox("Select Practice:", options=practice_options)

def update_graph(sub_location, selected_practice):
    selected_data = icb_data_raw_merged[(icb_data_raw_merged['sub_location'] == sub_location) & (icb_data_raw_merged['Practice'] == selected_practice)]
    selected_data = selected_data.groupby('date', as_index=False).agg({'Spend per 1000 Patients': 'sum'})
    selected_data['formatted_date'] = selected_data['date'].dt.strftime('%b %Y')

    pcn_code_values = icb_data_raw_merged[(icb_data_raw_merged['sub_location'] == sub_location) & (icb_data_raw_merged['Practice'] == selected_practice)]['PCN Code'].dropna().unique()
    pcn_code = pcn_code_values[0]
    pcn_practices = icb_data_raw_merged[(icb_data_raw_merged['PCN Code'] == pcn_code) & (icb_data_raw_merged['Practice'] != selected_practice)]
    pcn_practices = pcn_practices.groupby(['Practice', 'date'], as_index=False)['Spend per 1000 Patients'].sum()
    
    national_data = national_data_raw_merged[national_data_raw_merged['Country'] == 'ENGLAND']

    fig = go.Figure()

    for pcn_practice in pcn_practices['Practice'].unique():
        data = pcn_practices[pcn_practices['Practice'] == pcn_practice]
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['Spend per 1000 Patients'],
            mode='lines',
            line=dict(color='rgba(255, 99, 132, 0.2)', width=1),
            name=f"{pcn_practice} (same PCN)",
            hovertemplate=( 
                pcn_practice + '<br>' +
                'Spend per 1000 Patients: £%{y:.1f}<br>' +
                'Time Period: %{x|%b %Y}<extra></extra>' 
            ),
            showlegend=False
        ))

    fig.add_trace(go.Scatter(
        x=selected_data['date'],
        y=selected_data['Spend per 1000 Patients'],
        mode='lines',
        line=dict(color='firebrick', width=2),
        customdata=selected_data['formatted_date'],
        name=f"{selected_practice}",
        hovertemplate=(
            selected_practice + '<br>' +
            'Spend per 1000 Patients: £%{y:.1f}<br>' +
            'Time Period: %{customdata}<extra></extra>'
        ),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=national_data['date'],
        y=national_data['Spend per 1000 Patients'],
        mode='lines',
        name='National Average',
        customdata=national_data['formatted_date'],
        hovertemplate=(
            'National Average<br>' +
            'Spend per 1000 Patients: \£%{y:.1f}<br>' +
            'Time Period: %{customdata}<extra></extra>'
        ),
        line=dict(color='#2A6FBA', width=2),
        showlegend=False
    ))

    last_date = selected_data['date'].max()
    last_value = selected_data[selected_data['date'] == last_date]['Spend per 1000 Patients'].values[0]

    fig.add_annotation(
        x=last_date,
        y=last_value,
        text=selected_practice,
        showarrow=False,
        xanchor='left',
        yanchor='middle',
        font=dict(color='firebrick'),
        xshift=5
    )

    fig.update_layout(
        yaxis_title=f'{dataset_type} Spend per 1000 Patients',
        template='simple_white',
        height=700,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(b=100),
        yaxis_tickprefix="£"
    )

    last_nat_date = national_data['date'].max()
    last_nat_value = national_data[national_data['date'] == last_nat_date]['Spend per 1000 Patients'].values[0]

    fig.add_annotation(
        x=last_nat_date,
        y=last_nat_value,
        text='National Average',
        showarrow=False,
        xanchor='left',
        yanchor='middle',
        font=dict(color='#2A6FBA'),
        xshift=5
    )

    return fig

# Render the local trends chart
line_fig = update_graph(selected_sublocation, selected_practice)
st.plotly_chart(line_fig, use_container_width=True)
