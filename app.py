# app.py

import pandas as pd
import numpy as np
import calendar
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import scipy.stats as stats
import geopandas as gpd
import os
import requests
import json
from jupyter_dash import JupyterDash
import dash
import dash_extensions
from dash import Dash, html, dash_table, dcc, Input, Output, State, callback_context
from dash_extensions import EventListener
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set minimalist style for Plotly charts
pio.templates.default = 'simple_white'

# Add mapping for local SICBL names
sicbl_legend_mapping = {
    '84H': 'Durham',
    '00P': 'Sunderland',
    '00L': 'Northumberland',
    '01H': 'Cumbria',
    '13T': 'Newcastle-Gateshead',
    '16C': 'Tees Valley',
    '00N': 'North Tyneside',
    '99C': 'South Tyneside'
}

# Load data into dataframes
practice_data = pd.read_csv("pcn_practice_data.csv")
saba_data = pd.read_csv("sabas_by_practice_nenc.csv")

# Convert dates into standard datetime format
practice_data['date'] = pd.to_datetime(practice_data['date'], format='%Y%m')

# Convert practice names to Title case
practice_data['practice'] = practice_data['practice'].str.title()
saba_data['Practice'] = saba_data['Practice'].str.title()

# Strip dummy practices
saba_data = saba_data[saba_data['PCN'] != 'DUMMY'] 

# Remove the last 2 characters from the SICBL code
saba_data['Commissioner / Provider Code'] = saba_data['Commissioner / Provider Code'].str.slice(0, -2)

# Add formatted date column
practice_data['formatted_date'] = practice_data['date'].dt.strftime('%b %Y')

# Sum items and actual cost wherever practice code and date is the same
summary = (
    saba_data
    .groupby(['Practice Code', 'Year Month'], as_index=False)
    [['Items', 'Actual Cost']]
    .sum()
)

# Prepare base structure for reinsertion of summed data above (dropping duplicate rows to keep one representative row per group i.e. each unique practice / date combo)
base = (
    saba_data
    .drop_duplicates(subset=['Practice Code', 'Year Month'])
    .drop(columns=['Items', 'Actual Cost', 'BNF Chemical Substance'])
)

# Merge summed values with the base structure to create full data
saba_data_merged = pd.merge(base, summary, on=['Practice Code', 'Year Month'])

# Optionally assign a new BNF chemical code to make it clear which drugs the values refer to
saba_data_merged['BNF Chemical Substance Code'] = '0301011_MERGED'


# Step 1: Group and calculate mean values
means = (
    saba_data_merged
    .groupby('Practice Code', as_index=False)
    [['List Size', 'Actual Cost', 'Items']]
    .mean()
    .round(0)
)

# Step 2: Get one representative row per Practice Code (keeping original non-numeric info)
base = (
    saba_data_merged
    .drop_duplicates(subset='Practice Code')
    .drop(columns=['List Size', 'Actual Cost', 'Items', 'Year Month'])
)

# Step 3: Merge means back in
saba_means_merged = pd.merge(base, means, on='Practice Code')

saba_means_merged['Sub-location'] = saba_means_merged['Commissioner / Provider Code'].map(sicbl_legend_mapping)


# Calculate Spend per 1000 Patients for each Practice
saba_means_merged['Spend per 1000 Patients'] = ((saba_means_merged['Actual Cost'] / saba_means_merged['List Size'])*1000).round(1)

# Calculate Average Spend per 1000 Patients across ICB
total_actual_spend = saba_means_merged['Actual Cost'].sum()
total_list_size = saba_means_merged['List Size'].sum()
icb_average_spend = (total_actual_spend / total_list_size) * 1000

bar_dynamic = px.bar(
    saba_means_merged.sort_values('Spend per 1000 Patients', ascending=False),
    x='Practice',
    y='Spend per 1000 Patients',
    hover_name='Practice',
    #labels={'Actual Cost': 'Spend on SABAs'},
    hover_data={
        'Practice Code': False,
        'Sub-location': False,
        'Practice': False,
        'Spend per 1000 Patients': True,
        'Items': True,
    },
    color='Sub-location',
)
bar_dynamic.update_layout(
    height=700,
    width=1150,
    xaxis_title='',
    yaxis_title='',
    yaxis_tickprefix="£",
    legend_title_text=None,
    xaxis=dict(
        showticklabels=False,
        showgrid=False,
        tickangle=45,
        tickfont=dict(size=10)
    ),
    yaxis=dict(
        showgrid=False
    ),
    legend=dict(
        orientation="h",         # horizontal orientation
        yanchor="bottom",
        y=1,                 # moves it below the plot
        xanchor="center",
        x=0.5,                   # center the legend
        itemwidth=50,           # fixed width for each item
        itemsizing="trace",
        tracegroupgap=5,        # vertical spacing if wraps
    )
)

# Get the number of practices (to understand length of x-axis)
x_range = len(saba_means_merged['Practice Code'].unique())

# Add ICB average line
bar_dynamic.add_shape(
    type="line",
    x0=0,  # Start at far left of the plotting area
    x1=1,  # End at far right
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
)

## INITIALISE PYTHON DASH APP
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    style={
        'padding': '40px',
        'maxWidth': '1400px',
        'margin': '0 auto',
        'textAlign': 'center',
    },
    children=[
        html.Div(
            'Hormone Replacement Therapy (HRT) Dashboard',
            style={
                'color': 'linear-gradient(to right, #ff7e5f, #feb47b)',  # Gradient color for the text
                'fontSize': '40px',  # Increase font size for more impact
                'fontWeight': 'bold',  # Make the text bold
                'letterSpacing': '2px',  # Add spacing between letters for elegance
                'textShadow': '2px 2px 8px rgba(0, 0, 0, 0.1)',  # Add text shadow for depth
                'fontFamily': '"Helvetica Neue", Arial, sans-serif',  # Use a modern sans-serif font
                'marginBottom': '50px',  # Space below the title
                'padding': '10px',  # Add some padding around the title for better alignment
            }
        ),

        html.H3('Spend on SABAs: ICB-wide comparison in the last 3m', style={'marginTop': '40px'}),
        dcc.Graph(id='bar_dynamic_graph', figure=bar_dynamic),


        
        html.H3("Spend on SABAs: Local Trends"),
        html.Div(
                style={
                    'display': 'flex',          # Flexbox layout to position elements horizontally
                    'justifyContent': 'center', # Center the dropdowns horizontally
                    'gap': '20px',              # Add space between the dropdowns
                    'marginBottom': '30px'      # Add some bottom margin for spacing
                },
                children=[
                    html.Div([
                        html.Label("Select SICBL:"),
                        dcc.Dropdown(
                            id='sicbl-dropdown',
                            options=[{'label': sicbl_legend_mapping.get(sicbl, sicbl), 'value': sicbl} 
                                     for sicbl in sorted(practice_data['sicbl'].unique()) if sicbl != 'National'],
                            value=practice_data['sicbl'].unique()[0],
                            style={
                            'width': '350px',  # Increase width of dropdown
                            'fontSize': '16px', # Increase font size if needed
                            }
                        ),
                    ]),

                    html.Div([
                        html.Label("Select Practice:"),
                        dcc.Dropdown(
                            id='practice-dropdown',
                            style={
                            'width': '350px',  # Increase width of dropdown
                            'fontSize': '16px', # Increase font size if needed
                            }
                        ),
                    ])
                ]
            ),
        dcc.Graph(
            id='newlinegraph',
            style={'height': '600px'}
            ),
    ]
)

# Callbacks
@app.callback(
    Output("bar_dynamic_graph", "figure"),
    Input("bar_dynamic_graph", "restyleData"),
    State("bar_dynamic_graph", "figure"),
)
def update_bar_dynamic_graph_on_legend_toggle(restyle_data, fig):
    if not restyle_data:
        return fig

    # Here you can inspect visibility
    visibilities = []
    for i, trace in enumerate(fig["data"]):
        vis = trace.get("visible", True)
        if vis not in [False, "legendonly"]:
            visibilities.append(True)

    show_xticks = len(visibilities) == 1

    fig["layout"]["xaxis"]["showticklabels"] = show_xticks
    return fig

@app.callback(
    Output('practice-dropdown', 'options'),
    Output('practice-dropdown', 'value'),
    Input('sicbl-dropdown', 'value')
)
def set_practices(sicbl):
    filtered = practice_data[practice_data['sicbl'] == sicbl]
    options = [{'label': code, 'value': code} for code in sorted(filtered['practice'].unique())]
    value = options[0]['value'] if options else None
    return options, value

@app.callback(
    Output('newlinegraph', 'figure'),
    Input('sicbl-dropdown', 'value'),
    Input('practice-dropdown', 'value')
)
def update_graph(sicbl, selected_practice):
    if not selected_practice:
        return go.Figure()

    # Filter for selected practice
    selected_data = practice_data[
        (practice_data['sicbl'] == sicbl) & 
        (practice_data['practice'] == selected_practice)
    ]

    # Get the PCN of the selected practice
    pcn_code = selected_data['pcn_code'].iloc[0]

    # Get all practices in the same PCN
    pcn_practices = practice_data[
        (practice_data['pcn_code'] == pcn_code) & 
        (practice_data['practice'] != selected_practice)
    ]
    
    # Filter for national data
    national_data = practice_data[practice_data['region'] == 'National']

    # Start the figure
    fig = go.Figure()
    
    # Plot other practices in same PCN (pale red)
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
                'HRT Items per 1000 Patients: %{y:.1f}<br>' +
                'Time Period: %{customdata}<extra></extra>'
            ),
            showlegend=False
        ))

    # Plot selected practice (stronger line)
    fig.add_trace(go.Scatter(
        x=selected_data['date'],
        y=selected_data['hrt_items_per_patient'],
        mode='lines',
        line=dict(color='firebrick', width=2),
        customdata=selected_data['formatted_date'],
        name=f"{selected_practice}",
        hovertemplate=(
            selected_practice + '<br>' +
            'HRT Items per 1000 Patients: %{y:.1f}<br>' +
            'Time Period: %{customdata}<extra></extra>'
        ),
        showlegend=False
    ))
    
    # Add national average trace (ensure it appears in legend)
    fig.add_trace(go.Scatter(
        x=national_data['date'],
        y=national_data['hrt_items_per_patient'],
        mode='lines',
        name='National Average',
        customdata=selected_data['formatted_date'],
        hovertemplate=(
            'National Average<br>' +
            'HRT Items per 1000 Patients: %{y:.1f}<br>' +
            'Time Period: %{customdata}<extra></extra>'
        ),
        line=dict(color='#2A6FBA', width=2),
        showlegend=False
    ))

    # Add label at the end of selected practice line
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
        yaxis_title='SABA Spend per 1000 Patients',  #Could make SABA {workstream}
        template='simple_white'
    )

    # Add label at the end of national average line
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
    
    # Move legend to the bottom center
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=100)  # Add space at bottom to fit legend
    )

    return fig

# Run the app
if __name__ == "__main__":
    app.run(debug=True)