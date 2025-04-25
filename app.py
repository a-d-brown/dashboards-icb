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
from wordcloud import WordCloud
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
sicbl_data = pd.read_csv("hrt_by_sicbl.csv")
ethnicity_data = pd.read_csv("ethnicity.csv")
mosaic_data = pd.read_csv("mosaic_data.csv")
saba_data = pd.read_csv("sabas_by_practice_nenc.csv")

# Convert dates into standard datetime format
sicbl_data['date'] = pd.to_datetime(sicbl_data['date'], format='%d/%m/%Y')
practice_data['date'] = pd.to_datetime(practice_data['date'], format='%Y%m')
ethnicity_data['date'] = pd.to_datetime(ethnicity_data['date'], format='%d/%m/%Y')


# Convert practice names to Title case
practice_data['practice'] = practice_data['practice'].str.title()
saba_data['Practice'] = saba_data['Practice'].str.title()

# Strip dummy practices
saba_data = saba_data[saba_data['PCN'] != 'DUMMY'] 


# Remove the last 2 characters from the SICBL code
saba_data['Commissioner / Provider Code'] = saba_data['Commissioner / Provider Code'].str.slice(0, -2)


# Add formatted date column
practice_data['formatted_date'] = practice_data['date'].dt.strftime('%b %Y')


# Define latest data as a slice for convenient access later
jan25_data = practice_data[practice_data['date'] == pd.Timestamp('2025-01-01')]

# Map each SICBL to its display name and add as a new column
jan25_data.loc[:, 'sicbl_display'] = jan25_data['sicbl'].map(sicbl_legend_mapping)

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


## CHLOROPLETH

# Download GeoSpatial mapping data as raw GeoJSON
url = 'https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/Sub_Integrated_Care_Board_Locations_April_2023_EN_BGC/FeatureServer/0/query?outFields=*&where=1=1&f=geojson'
response = requests.get(url)
data = response.json()

# Save to file
with open('sicbl_geojson.geojson', 'w') as f:
    json.dump(data, f)

# Load it into a dataframe using GeoPandas
sicbl_gdf = gpd.read_file('sicbl_geojson.geojson').to_crs(epsg=4326)

# Derive a new column containing only the 3 character CCG codes from the SICBL23NM column 
sicbl_gdf['ccg_code'] = sicbl_gdf['SICBL23NM'].str.split().str[-1]

# Join geodata with patient data
hrt_chloropleth_data = sicbl_data.merge(sicbl_gdf, left_on='sicbl', right_on='ccg_code', how='inner')

# Convert df to a geodataframe to ensure geometry column is recognised in chloropleth code
hrt_chloropleth_data = gpd.GeoDataFrame(hrt_chloropleth_data, geometry='geometry')

# Rename sicbl_name as Location for improved hover over
hrt_chloropleth_data = hrt_chloropleth_data.rename(columns={"sicbl_name": "Location"})

# Create copy of Location to set as index (because index columns can't be referred to for hover)
hrt_chloropleth_data["Location_copy"] = hrt_chloropleth_data["Location"]
hrt_chloropleth_data = hrt_chloropleth_data.set_index("Location_copy")

# Round pct_patients to 1dp, then rename column for improved hover
hrt_chloropleth_data["pct_patient"] = hrt_chloropleth_data["pct_patient"].round(1)
hrt_chloropleth_data["Percentage"] = hrt_chloropleth_data["pct_patient"].astype(str) + "%"

# Plotly Express choropleth map
chloropleth = px.choropleth(hrt_chloropleth_data,
                    geojson=hrt_chloropleth_data.geometry.__geo_interface__,
                    locations="Location", 
                    color="pct_patient",
                    color_continuous_scale="Blues",
                    range_color=(0, hrt_chloropleth_data['pct_patient'].max()),
                    labels={"pct_patient": ""},
                    hover_name='Location',
                    hover_data={"Percentage": True, "pct_patient": False, "Location": False}
                   )


# Update layout for better aesthetics
chloropleth.update_geos(
    showcoastlines=False,
    coastlinecolor="black",
    showland=True,
    landcolor="white", 
    projection_type="mercator",
    center={"lat": 53.29851172418648, "lon": -0.9046802336339},
    projection_scale=30,
    showframe=False
)

chloropleth.update_layout(
    margin={"r": 0, "t": 40, "l": 20, "b": 0},  # Adjust margins to remove extra space
    autosize=True,
    coloraxis_colorbar=dict(
        len=0.8,
        x=0.64,  # Position colorbar closer to the map
        xanchor="left",
        y=0.5,  # Adjust vertical position of colorbar
        yanchor="middle",
        title_side="right",  # Move title to the right of the color bar
        ticktext=['0%', '0.7%', '1.4%', '2.1%', '2.8%'],  # labels to display
    ),
    width=1200,  # Set the width of the map
    height=800  # Set the height of the map
)

# Add custom color bar ticks
chloropleth.update_coloraxes(colorbar_ticksuffix=" items", colorbar_tickvals=[0, 0.7, 1.4, 2.1, 2.8])


## LINE GRAPH

# Plot the line graph
linegraph = px.line(
    ethnicity_data,
    x='date',
    y='rate_per_1000',
    color='ethnicity',
    labels={'rate_per_1000': 'HRT items per 1000 patients', 'date': 'Date'},
    hover_name='ethnicity',
    hover_data={'ethnicity': False, 'date': True, 'rate_per_1000': True}  # Only display 'prevalence' in hover, remove 'ethnicity'
)

linegraph.update_layout(
    xaxis_title='',
    yaxis_title='HRT items per 1000 patients',
    legend_title='Ethnicity',
    width=1300,
    height=700
)

linegraph.add_vline(
    x=pd.Timestamp('03-30-2020'),
    line_dash='dash',
    line_color='black',
    line_width=2,
)

linegraph.add_annotation(
    x=pd.Timestamp('03-30-2020') + pd.Timedelta(days=1),
    y=ethnicity_data['rate_per_1000'].max(),
    text="Start of COVID-19 lockdown",
    showarrow=True,
    arrowhead=1,
    yshift=10,
    ax=50,
    xanchor='left'  # Anchor the text to the left of the annotation point

)

linegraph.update_traces(line=dict(width=1.5))



## SCATTER PLOT

# Round data to 1dp
jan25_data["antidepressant_items_per_patient"] = jan25_data["antidepressant_items_per_patient"].round(1)
jan25_data["hrt_items_per_patient"] = jan25_data["hrt_items_per_patient"].round(1)


scatterplot = px.scatter(
    jan25_data,
    x='antidepressant_items_per_patient',
    y='hrt_items_per_patient',
    color='sicbl_display',
    size='list_size',
    hover_name='practice',
    labels={
        'hrt_items_per_patient': 'HRT items per 1000 patients',
        'antidepressant_items_per_patient': 'Antidepressant items per 1000 patients',
        'sicbl_display': 'Sub-location',
        'list_size': 'Registered Patients'
    },
)

scatterplot.update_layout(width=1300, height=700)

## ETHNICITY BAR CHART

# Sort data as descending
mosaic_data_sorted = mosaic_data.sort_values(by='pct_on_hrt', ascending=False)

# Divide by 100 because Plotly likes percentages to be expressed as decimals
mosaic_data_sorted['pct_on_hrt'] = mosaic_data_sorted['pct_on_hrt'] / 100


# Create the bar chart
ethnicity_bars = px.bar(mosaic_data_sorted, 
             x='ethnicity', 
             y='pct_on_hrt', 
             hover_name='ethnicity',  # Display only the 'ethnicity' in hover
             hover_data={'ethnicity': False, 'prevalence': True})  # Only display 'prevalence' in hover, remove 'ethnicity'

# Define colours based on conditional value in y-axis
colors = ['lightgray' if val > 0.0215 else '#DC143C' for val in mosaic_data_sorted['pct_on_hrt']]

ethnicity_bars.update_traces(marker_color=colors)

ethnicity_bars.update_traces(hovertemplate="<b>%{x}</b><br>Patients in Ethnic Group taking HRT: %{y:.1%}<br>Representation of Ethnic Group in Local Population: %{customdata[0]}%")

# Add a horizontal line at y = 2.49
ethnicity_bars.add_shape(
    type="line", 
    x0=-0.5, x1=len(mosaic_data_sorted)-0.5,  # Span the line across the x-axis range
    y0=0.0215, y1=0.0215, 
    line=dict(color="red", width=2, dash="dash")
)

# Add a label on the right-hand side of the horizontal line
ethnicity_bars.add_annotation(
    x=len(mosaic_data_sorted)-0.166,  # Position at the right end of the x-axis
    y=0.0215,
    text="National Average",
    showarrow=False,
    font=dict(size=12, color="red"),
    align="left"
)

ethnicity_bars.update_layout(
    width=1200,  # Increase width (default is 700)
    height=800,  # Increase height (default is 400)
    yaxis_tickformat="0.1%",
    yaxis_title='',  # Remove y-axis title
    xaxis_title='',  # Remove x-axis title
    hoverlabel=dict(
        bgcolor='rgba(255, 255, 255, 0.8)',  # Set background color of the hover box (semi-transparent white)
        font_size=12,  # Font size for hover text
        font_family="Arial",  # Font family
        font_color="black"  # Font color (text in the hover box)
    ),
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

        html.H3('Percentage of women aged 40-54 years in the North East taking HRT by ethnic group in Jan25', style={'marginTop': '40px'}),
        dcc.Graph(id='bar_dynamic_graph', figure=bar_dynamic),


        
        html.H3("HRT prescribing in women aged 40-54y: GP Practice Trends"),
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
        
        html.H3('HRT prescribing in women aged 40-54y in the North East by ethnic group ', style={'marginTop': '40px'}),
        dcc.Graph(figure=linegraph),

        html.H3('Antidepressant and HRT prescribing rates in women aged 40-54y across GP Practices in the North East in Jan25', style={'marginTop': '40px'}),
        dcc.Graph(figure=scatterplot),
        
        html.H3('Percentage of women aged 40-54 years taking HRT across England in Jan25', style={'marginTop': '40px'}),
        dcc.Graph(figure=chloropleth),
        
        html.H3('Percentage of women aged 40-54 years in the North East taking HRT by ethnic group in Jan25', style={'marginTop': '40px'}),
        dcc.Graph(figure=ethnicity_bars),
        
        html.H3('Barriers to HRT Access: Clinician vs Patient Perceptions', style={'marginTop': '40px'}),
        html.Img(
            src='/assets/wordcloud.png',
            style={
                'width': '75%',
                'marginTop': '10px',
                'borderRadius': '10px',
                'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)'
            }
        )
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
        title=f"HRT Items per 1000 Patients for {selected_practice} and PCN Peers",
        yaxis_title='HRT Items per 1000 Patients',
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