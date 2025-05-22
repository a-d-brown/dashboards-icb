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

# Add reverse mapping for SICBL names back to codes
sicbl_reverse_mapping = {v: k for k, v in sicbl_legend_mapping.items()}

# Define fixed colors for Sub-locations
sub_location_colors = {
    'Durham': '#1f77b4',          # deeper blue
    'Sunderland': '#d62728',      # richer red
    'Northumberland': '#2ca02c',  # vivid green
    'Cumbria': '#9467bd',         # stronger purple
    'Newcastle-Gateshead': '#ff7f0e',  # bold orange
    'Tees Valley': '#17becf',     # clearer cyan
    'North Tyneside': '#e377c2',  # bright pink
    'South Tyneside': '#bcbd22',  # rich olive
}

# Load data into dataframes
saba_line_chart_data = pd.read_csv("pcn_practice_data.csv") # will need to load all measure data immediately
saba_bar_chart_data = pd.read_csv("sabas_by_practice_nenc.csv")

# Data preprocessing
saba_line_chart_data['date'] = pd.to_datetime(saba_line_chart_data['date'], format='%Y%m')
saba_line_chart_data['practice'] = saba_line_chart_data['practice'].str.title()
saba_bar_chart_data['Practice'] = saba_bar_chart_data['Practice'].str.title()
saba_bar_chart_data = saba_bar_chart_data[saba_bar_chart_data['PCN'] != 'DUMMY']
saba_bar_chart_data['Commissioner / Provider Code'] = saba_bar_chart_data['Commissioner / Provider Code'].str.slice(0, -2)
saba_line_chart_data['formatted_date'] = saba_line_chart_data['date'].dt.strftime('%b %Y')

# Summarize SABA data
summary = saba_bar_chart_data.groupby(['Practice Code', 'Year Month'], as_index=False)[['Items', 'Actual Cost']].sum()
base = saba_bar_chart_data.drop_duplicates(subset=['Practice Code', 'Year Month']).drop(columns=['Items', 'Actual Cost', 'BNF Chemical Substance'])
saba_bar_chart_data_merged = pd.merge(base, summary, on=['Practice Code', 'Year Month'])
saba_bar_chart_data_merged['BNF Chemical Substance Code'] = '0301011_MERGED'

# Calculate means
means = saba_bar_chart_data_merged.groupby('Practice Code', as_index=False)[['List Size', 'Actual Cost', 'Items']].mean().round(0)
base = saba_bar_chart_data_merged.drop_duplicates(subset='Practice Code').drop(columns=['List Size', 'Actual Cost', 'Items', 'Year Month'])
saba_means_merged = pd.merge(base, means, on='Practice Code')
saba_means_merged['Sub-location'] = saba_means_merged['Commissioner / Provider Code'].map(sicbl_legend_mapping)
saba_means_merged['Spend per 1000 Patients'] = ((saba_means_merged['Actual Cost'] / saba_means_merged['List Size'])*1000).round(1)

# ICB Average Spend
total_actual_spend = saba_means_merged['Actual Cost'].sum()
total_list_size = saba_means_merged['List Size'].sum()
icb_average_spend = (total_actual_spend / total_list_size) * 1000

# Streamlit Layout
st.title("NENC Medicines Optimisation Workstream Dashboard")


## Bar Chart Comparison
st.header('Spend on SABAs: ICB-wide comparison in the last 3m')   #will refactor to draw text and datafile from list of {dropdowns later}

# Initialize session state for toggles
for subloc in sub_location_colors.keys():
    if f"subloc_toggle_{subloc}" not in st.session_state:
        st.session_state[f"subloc_toggle_{subloc}"] = True

# --- Toggle UI ---
# Create a column for each sublocation
cols = st.columns(len(sub_location_colors))
rerun_needed = False

for i, (subloc, color) in enumerate(sub_location_colors.items()):
    with cols[i]:
        # Use the first row for the color swatch and sublocation name
        row_1 = st.columns([0.15, 0.5])  # [color swatch, sublocation text]

        # Determine if the sublocation is toggled off (strikethrough and greyed out)
        is_toggled_off = not st.session_state.get(f"subloc_toggle_{subloc}", False)

        # Apply strikethrough and grey out style if toggled off
        strikethrough_style = "text-decoration: line-through; color: #d3d3d3;" if is_toggled_off else ""

        # Color swatch with reduced margin to bring it closer to the text
        row_1[0].markdown(
            f"""
            <div style='
                width: 16px;
                height: 16px;
                background-color: {color};
                border-radius: 3px;
                margin-right: 0px;
                border: 1px solid #00000033;
            '></div>
            """,
            unsafe_allow_html=True
        )

        # Sublocation name with optional strikethrough and reduced font size
        row_1[1].markdown(
            f"""<div title="{subloc}" style='
                font-weight: bold;
                font-size: 13px;
                cursor: default;
                vertical-align: top;
                text-align: left;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                max-width: 12vw;
                min-width: 60px;
                {strikethrough_style}'>
                {subloc}
            </div>""",
            unsafe_allow_html=True
        )

        # Use the second row for the toggle button
        row_2 = st.columns([1])  # [toggle button in a single column]

        # Toggle button for each sublocation (below the text)
        if row_2[0].button(f"Toggle", key=f"toggle_{subloc}"):
            # Toggle the session state for the sublocation
            st.session_state[f"subloc_toggle_{subloc}"] = not st.session_state.get(f"subloc_toggle_{subloc}", False)
            rerun_needed = True

if rerun_needed:
    st.rerun()




# Collect selected Sub-locations
selected_sublocations = [
    subloc for subloc in sub_location_colors.keys()
    if st.session_state[f"subloc_toggle_{subloc}"]
]

# Filter data based on selected Sub-locations
filtered_data = saba_means_merged[saba_means_merged['Sub-location'].isin(selected_sublocations)]

# Determine if x-axis tick labels should be shown
show_xticks = len(selected_sublocations) == 1

# Create updated bar chart
bar_dynamic = px.bar(
    filtered_data.sort_values('Spend per 1000 Patients', ascending=False),
    x='Practice',
    y='Spend per 1000 Patients',
    hover_name='Practice',
    hover_data={'Practice Code': False, 'Sub-location': False, 'Practice': False, 'Spend per 1000 Patients': True, 'Items': True},
    color='Sub-location',
    color_discrete_map=sub_location_colors,
)

bar_dynamic.update_layout(
    height=700,
    width=1150,
    xaxis_title='',
    yaxis_title='SABA Spend per 1000 Patients',
    yaxis_tickprefix="£",
    legend_title_text=None,
    xaxis=dict(
        showticklabels=show_xticks,
        showgrid=False,
        tickangle=45,
        tickfont=dict(size=10)
    ),
    yaxis=dict(showgrid=False),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="center",
        x=0.5,
        itemwidth=50,
        itemsizing="trace",
        tracegroupgap=5,
    ),
    showlegend=False,
)

bar_dynamic.update_traces(width=0.6)

# Add ICB average line
bar_dynamic.add_shape(
    type="line",
    x0=0,
    x1=1,
    y0=icb_average_spend,
    y1=icb_average_spend,
    line=dict(color="black", width=1.5, dash="dash"),
    xref="paper",
    yref="y"
)

bar_dynamic.add_annotation(
    x=1,
    y=icb_average_spend,
    text="ICB Average",
    showarrow=False,
    font=dict(size=12, color="black"),
    xref="paper",
    yref="y",
    xanchor="left",
    yanchor="middle"
)

bar_dynamic.update_layout(
    margin=dict(r=100)  # Increase the right margin to allow space for annotation
)

# Display the chart
st.plotly_chart(bar_dynamic, use_container_width=True)



st.markdown('<hr class="divider">', unsafe_allow_html=True)


## --- Local Trends Section ---

st.header("Spend on SABAs: Local Trends")

# Create two columns for the dropdowns to be displayed side by side
col1, col2 = st.columns(2)

with col1:
    # SICBL dropdown
    sicbl_options = [sicbl_legend_mapping.get(sicbl, sicbl) for sicbl in sorted(saba_line_chart_data['sicbl'].unique()) if sicbl != 'National']
    selected_sicbl = st.selectbox("Select Sub-location:", options=sicbl_options)

with col2:
    # Practice dropdown
    selected_sicbl_code = sicbl_reverse_mapping[selected_sicbl]  # Map back to code
    filtered_practices = saba_line_chart_data[saba_line_chart_data['sicbl'] == selected_sicbl_code]
    practice_options = sorted(filtered_practices['practice'].unique())
    selected_practice = st.selectbox("Select Practice:", options=practice_options)



# Interactive Graph Function
def update_graph(sicbl_code, selected_practice):
    selected_data = saba_line_chart_data[(saba_line_chart_data['sicbl'] == sicbl_code) & (saba_line_chart_data['practice'] == selected_practice)]

    if selected_data.empty:
        st.warning("No data available for the selected SICBL and practice.")
        return go.Figure()  # Return an empty figure to avoid breaking Streamlit

    pcn_code = selected_data['pcn_code'].iloc[0]
    pcn_practices = saba_line_chart_data[(saba_line_chart_data['pcn_code'] == pcn_code) & (saba_line_chart_data['practice'] != selected_practice)]
    national_data = saba_line_chart_data[saba_line_chart_data['sicbl'] == 'National']

    fig = go.Figure()

    for pcn_practice in pcn_practices['practice'].unique():
        data = pcn_practices[pcn_practices['practice'] == pcn_practice]
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['hrt_items_per_patient'],
            mode='lines',
            line=dict(color='rgba(255, 99, 132, 0.2)', width=1),
            customdata=selected_data['formatted_date'],
            name=f"{pcn_practice} (same PCN)",
            hovertemplate=(
                pcn_practice + '<br>' +
                'Spend per 1000 Patients: £%{y:.1f}<br>' +
                'Time Period: %{customdata}<extra></extra>'
            ),
            showlegend=False
        ))

    fig.add_trace(go.Scatter(
        x=selected_data['date'],
        y=selected_data['hrt_items_per_patient'],
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
        y=national_data['hrt_items_per_patient'],
        mode='lines',
        name='National Average',
        customdata=selected_data['formatted_date'],
        hovertemplate=(
            'National Average<br>' +
            'Spend per 1000 Patients: £%{y:.1f}<br>' +
            'Time Period: %{customdata}<extra></extra>'
        ),
        line=dict(color='#2A6FBA', width=2),
        showlegend=False
    ))

    last_date = selected_data['date'].max()
    last_value = selected_data[selected_data['date'] == last_date]['hrt_items_per_patient'].values[0]
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
        yaxis_title='SABA Spend per 1000 Patients',
        template='simple_white',
        height=700
    )

    last_nat_date = national_data['date'].max()
    last_nat_value = national_data[national_data['date'] == last_nat_date]['hrt_items_per_patient'].values[0]

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

    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(b=100),
         yaxis_tickprefix="£"
    )

    return fig

# Render the local trends chart
line_fig = update_graph(selected_sicbl_code, selected_practice)
st.plotly_chart(line_fig, use_container_width=True)