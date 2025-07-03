import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Define function to plot bar chart
def plot_icb_bar_chart(filtered_data, measure_type, sub_location_colors, icb_average_value,
                       dataset_type, measure_metadata, selected_practice=None, delta_column=None, y_axis_label=None):

    show_xticks = filtered_data['sub_location'].nunique() == 1

    filtered_data = filtered_data.sort_values(measure_type, ascending=False)
    filtered_data = filtered_data.copy()
    filtered_data[measure_type] = filtered_data[measure_type]

    fig = px.bar(
        filtered_data,
        x='Practice',
        y=measure_type,
        color='sub_location',
        color_discrete_map=sub_location_colors
    )

    # Determine whether to add £ prefix
    if measure_type in measure_metadata:
        prefix = measure_metadata[measure_type].get("prefix", "")
        suffix = measure_metadata[measure_type].get("suffix", "")
    else:
        prefix = ""
        suffix = ""



    # Add formatted hovertemplate
    for trace in fig.data:
        subloc = trace.name
        subloc_data = filtered_data[filtered_data['sub_location'] == subloc]

        # Build customdata array depending on delta_column
        if delta_column and delta_column in subloc_data.columns:
            trace.customdata = subloc_data[["Items (monthly average)", delta_column]].to_numpy()
            trace.hovertemplate = (
                f"<b>%{{x}}</b><br>{measure_type}: {prefix}%{{y:.1f}}{suffix}<br>"
                "Items (monthly average): %{customdata[0]:.0f}<extra></extra>"
            )
        else:
            trace.customdata = subloc_data[["Items (monthly average)"]].to_numpy()
            trace.hovertemplate = (
                f"<b>%{{x}}</b><br>{measure_type}: {prefix}%{{y:.1f}}<br>"
                "Items (monthly average): %{customdata[0]:.0f}<extra></extra>"
            )

        trace.width = 0.6



    # Highlight logic
    if selected_practice:
        try:
            target_pcn = filtered_data[filtered_data['Practice'] == selected_practice]['PCN'].values[0]
        except IndexError:
            target_pcn = None

        fig.for_each_trace(lambda trace: _recolor_bars(
            trace, filtered_data, selected_practice, target_pcn, sub_location_colors
        ))

    # Dynamically scale y-axis for delta vs rate
    if measure_type in ["Δ from previous", "Δ from last year"]:
        data_max = filtered_data[measure_type].max()
        y_max = max(data_max * 1.05, 5)  # Add 10% margin or minimum 5
        y_min = min(filtered_data[measure_type].min(), -5)
        yaxis_range = [y_min, y_max]
    else:
        yaxis_range = None


    fig.update_layout(
        height=700,
        width=1150,
        xaxis_title='',
        yaxis_title = y_axis_label or f'{dataset_type} {measure_type}',
        yaxis_tickprefix=prefix,
        yaxis_ticksuffix=suffix,
        legend_title_text=None,
        xaxis=dict(showticklabels=show_xticks, showgrid=False, tickangle=45, tickfont=dict(size=10)),
        yaxis=dict(showgrid=False, range=yaxis_range),
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5),
        margin=dict(r=100),
        showlegend=False
    )

    # Add ICB average line (unless showing % CHANGE)
    if measure_type not in ["Δ from previous", "Δ from last year"]:
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

# Helper function used inside the bar chart main function
def _recolor_bars(trace, df, selected_practice, target_pcn, sub_location_colors):
    if 'x' in trace and hasattr(trace.marker, 'color'):
        current_colors = list(trace.marker.color) if isinstance(trace.marker.color, list) else [trace.marker.color] * len(trace.x)
        new_colors = []

        for i, practice in enumerate(trace.x):
            if practice == selected_practice:
                new_colors.append('#FF2D55')  # neon pink
            elif target_pcn and df[df['Practice'] == practice]['PCN'].values[0] == target_pcn:
                new_colors.append('#FFB3C6')  # neon aquamarine
            else:
                new_colors.append('#DDDDDD')  # light grey for all others

        trace.marker.color = new_colors


# Define function to plot line chart
def plot_line_chart(icb_data_raw_merged, national_data_raw_merged, sub_location, selected_sublocation, selected_practice, measure_type, dataset_type, mode="practice", sub_location_colors=None):
    fig = go.Figure()

    if mode == "sublocations":
        avg_by_subloc = (
            icb_data_raw_merged
            .groupby(['sub_location', 'date'])[measure_type]
            .mean()
            .reset_index()
        )

        unique_sublocs = avg_by_subloc['sub_location'].dropna().unique()

        for subloc in unique_sublocs:
            subloc_data = avg_by_subloc[avg_by_subloc['sub_location'] == subloc]

            if selected_sublocation == "Show all":
                # Use full color scheme with uniform width
                color = sub_location_colors.get(subloc, '#cccccc')
                width = 2
            else:
                # Highlight selected, fade others
                color = '#FF2D55' if subloc == selected_sublocation else '#FFD0DC'
                width = 2 if subloc == selected_sublocation else 1

            fig.add_trace(go.Scatter(
                x=subloc_data['date'],
                y=subloc_data[measure_type],
                mode='lines',
                name=subloc,
                line=dict(color=color, width=width),
                hovertemplate=f"{subloc}<br>Value: %{{y:.1f}}<br>Date: %{{x|%b %Y}}<extra></extra>",
                showlegend=False
            ))

        # Annotate selected sublocation if not "Show all"
        if selected_sublocation != "Show all":
            selected_line = avg_by_subloc[avg_by_subloc['sub_location'] == selected_sublocation]
            last_date = selected_line['date'].max()
            last_value = selected_line[selected_line['date'] == last_date][measure_type].values[0]

            fig.add_annotation(
                x=last_date,
                y=last_value,
                text=selected_sublocation,
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                font=dict(color='firebrick'),
                xshift=5
            )

            national_data = national_data_raw_merged[national_data_raw_merged['Country'] == 'ENGLAND']

            fig.add_trace(go.Scatter(
                x=national_data['date'],
                y=national_data[measure_type],
                mode='lines',
                name='National Average',
                customdata=national_data['date'].dt.strftime('%b %Y'),
                hovertemplate=(
                    'National Average<br>' +
                    'Value: %{y:.1f}<br>' +
                    'Time Period: %{customdata}<extra></extra>'
                ),
                line=dict(color='#4A90E2', width=2),
                showlegend=False
            ))

            # Label the last national value
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
            xaxis_title='',
            yaxis_title=f'{dataset_type} {measure_type}',
            template='simple_white',
            height=700,
            yaxis_tickprefix="£" if "Spend" in measure_type else "",
            margin=dict(b=100)
        )

        return fig

    # Prepare selected practice data
    selected_data = icb_data_raw_merged[
        (icb_data_raw_merged['sub_location'] == sub_location) &
        (icb_data_raw_merged['Practice'] == selected_practice)
    ].groupby('date', as_index=False).agg({measure_type: 'sum'})
    selected_data['formatted_date'] = selected_data['date'].dt.strftime('%b %Y')

    # Get PCN code for selected practice
    pcn_code_values = icb_data_raw_merged[
        (icb_data_raw_merged['sub_location'] == sub_location) &
        (icb_data_raw_merged['Practice'] == selected_practice)
    ]['PCN Code'].dropna().unique()
    pcn_code = pcn_code_values[0] if len(pcn_code_values) > 0 else None

    # Get data for other practices in the same PCN
    pcn_practices = icb_data_raw_merged[
        (icb_data_raw_merged['PCN Code'] == pcn_code) &
        (icb_data_raw_merged['Practice'] != selected_practice)
    ].groupby(['Practice', 'date'], as_index=False)[measure_type].sum()

    # National-level data
    national_data = national_data_raw_merged[national_data_raw_merged['Country'] == 'ENGLAND']

    # Plot other PCN practices
    for pcn_practice in pcn_practices['Practice'].unique():
        data = pcn_practices[pcn_practices['Practice'] == pcn_practice]
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data[measure_type],
            mode='lines',
            line=dict(color='#FFD0DC', width=1),
            name=f"{pcn_practice} (same PCN)",
            hovertemplate=(
                pcn_practice + '<br>' +
                'Value: %{y:.1f}<br>' +
                'Time Period: %{x|%b %Y}<extra></extra>'
            ),
            showlegend=False
        ))

    # Plot selected practice
    fig.add_trace(go.Scatter(
        x=selected_data['date'],
        y=selected_data[measure_type],
        mode='lines',
        line=dict(color='#FF2D55', width=2),
        customdata=selected_data['formatted_date'],
        name=f"{selected_practice}",
        hovertemplate=(
            selected_practice + '<br>' +
            'Value: %{y:.1f}<br>' +
            'Time Period: %{customdata}<extra></extra>'
        ),
        showlegend=False
    ))

    # Plot national average
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
        line=dict(color='#4A90E2', width=2),
        showlegend=False
    ))

    # Label last value for selected practice
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

    # Label last national value
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

    # Final layout tweaks
    fig.update_layout(
        yaxis_title=f'{dataset_type} {measure_type}',
        template='simple_white',
        height=700,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(b=100),
        yaxis_tickprefix="£" if "Spend" in measure_type else "",
    )

    return fig


def plot_high_cost_drugs_scatter(hcd_df, selected_practice):
    # Filter to selected practice
    df = hcd_df[hcd_df["Practice"] == selected_practice].copy()

    if df.empty:
        return None  # graceful fail if no data

    # Aggregate by BNF Presentation plus Code
    summary = (
        df.groupby("BNF Presentation plus Code", as_index=False)[["Items", "Actual Cost"]]
        .sum()
        .sort_values("Actual Cost", ascending=False)
        .head(100)
    )

    # Explicitly assign the presentation name to a label column (in case hover_name fails)
    summary["Label"] = summary["BNF Presentation plus Code"]

    # Generate scatter plot
    fig = px.scatter(
        summary,
        x="Items",
        y="Actual Cost",
        hover_name="Label",
        labels={"Actual Cost": "Actual Cost (£)", "Items": "Items"}
    )

    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(height=700)

    return fig


