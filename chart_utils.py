import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Define function to plot bar chart
def plot_icb_bar_chart(filtered_data, measure_type, sub_location_colors, icb_average_value, dataset_type):
    show_xticks = filtered_data['sub_location'].nunique() == 1

    fig = px.bar(
        filtered_data.sort_values(measure_type, ascending=False),
        x='Practice',
        y=measure_type,
        hover_name='Practice',
        hover_data={
            'sub_location': False,
            'Practice': False,
            measure_type: True,
            'Items (monthly average)': True
        },
        color='sub_location',
        color_discrete_map=sub_location_colors,
    )

    fig.update_layout(
        height=700,
        width=1150,
        xaxis_title='',
        yaxis_title=f'{dataset_type} {measure_type}',
        yaxis_tickprefix="£" if measure_type == 'Spend per 1000 Patients' else "",
        legend_title_text=None,
        xaxis=dict(showticklabels=(len(filtered_data['sub_location'].unique()) == 1), showgrid=False, tickangle=45, tickfont=dict(size=10)),
        yaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5),
        showlegend=False,
        margin=dict(r=100)
    )

    fig.update_traces(width=0.6)

    # Add ICB average line
    fig.add_shape(
        type="line", x0=0, x1=1, y0=icb_average_value, y1=icb_average_value,
        line=dict(color="black", width=1.5, dash="dash"), xref="paper", yref="y"
    )

    fig.add_annotation(
        x=1, y=icb_average_value,
        text="ICB Average",
        showarrow=False,
        font=dict(size=12, color="black"),
        xref="paper", yref="y",
        xanchor="left", yanchor="middle"
    )

    return fig


# Define function to plot line chart
def plot_line_chart(icb_data_raw_merged, national_data_raw_merged, sub_location, selected_practice, measure_type, dataset_type):
    selected_data = icb_data_raw_merged[
        (icb_data_raw_merged['sub_location'] == sub_location) &
        (icb_data_raw_merged['Practice'] == selected_practice)
    ].groupby('date', as_index=False).agg({measure_type: 'sum'})
    selected_data['formatted_date'] = selected_data['date'].dt.strftime('%b %Y')

    pcn_code_values = icb_data_raw_merged[
        (icb_data_raw_merged['sub_location'] == sub_location) &
        (icb_data_raw_merged['Practice'] == selected_practice)
    ]['PCN Code'].dropna().unique()
    pcn_code = pcn_code_values[0] if len(pcn_code_values) > 0 else None

    pcn_practices = icb_data_raw_merged[
        (icb_data_raw_merged['PCN Code'] == pcn_code) &
        (icb_data_raw_merged['Practice'] != selected_practice)
    ].groupby(['Practice', 'date'], as_index=False)[measure_type].sum()

    national_data = national_data_raw_merged[national_data_raw_merged['Country'] == 'ENGLAND']

    fig = go.Figure()

    for pcn_practice in pcn_practices['Practice'].unique():
        data = pcn_practices[pcn_practices['Practice'] == pcn_practice]
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data[measure_type],
            mode='lines',
            line=dict(color='rgba(255, 99, 132, 0.2)', width=1),
            name=f"{pcn_practice} (same PCN)",
            hovertemplate=(
                pcn_practice + '<br>' +
                'Value: %{y:.1f}<br>' +
                'Time Period: %{x|%b %Y}<extra></extra>'
            ),
            showlegend=False
        ))

    fig.add_trace(go.Scatter(
        x=selected_data['date'],
        y=selected_data[measure_type],
        mode='lines',
        line=dict(color='firebrick', width=2),
        customdata=selected_data['formatted_date'],
        name=f"{selected_practice}",
        hovertemplate=(
            selected_practice + '<br>' +
            'Value: %{y:.1f}<br>' +
            'Time Period: %{customdata}<extra></extra>'
        ),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=national_data['date'],
        y=national_data[measure_type],
        mode='lines',
        name='National Average',
        customdata=national_data['formatted_date'],
        hovertemplate=(
            'National Average<br>' +
            'Value: %{y:.1f}<br>' +
            'Time Period: %{customdata}<extra></extra>'
        ),
        line=dict(color='#2A6FBA', width=2),
        showlegend=False
    ))

    last_date = selected_data['date'].max()
    last_value = selected_data[selected_data['date'] == last_date][measure_type].values[0]
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

    last_nat_date = national_data['date'].max()
    last_nat_value = national_data[national_data['date'] == last_nat_date][measure_type].values[0]
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
        yaxis_title=f'{dataset_type} {measure_type}',
        template='simple_white',
        height=700,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(b=100),
        yaxis_tickprefix="£" if measure_type == 'Spend per 1000 Patients' else "",
    )

    return fig