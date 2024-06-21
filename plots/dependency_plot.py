from bokeh.plotting import figure
from bokeh.models import Band, ColumnDataSource, HoverTool, Legend, LegendItem
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import bokeh.colors
from calculations.similarity import get_similar_items


# for the contours
def kde(x, y, N):
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    X, Y = np.mgrid[xmin:xmax:N * 1j, ymin:ymax:N * 1j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    return X, Y, Z


def dependency_scatterplot(data, col, all_selected_cols, item, chart_type):
    relative = True
    item_style = "line"
    add_clusters = False
    sorted_data = data.copy().sort_values(by=col)
    mean = data[item.predict_class].mean()
    if relative:
        sorted_data[item.predict_class] = sorted_data[item.predict_class].apply(lambda x: x- mean)


    x_range = (sorted_data[col].min(), sorted_data[col].max())
    y_range = [np.floor(sorted_data[item.predict_class].min()), np.ceil(sorted_data[item.predict_class].max())]

    if (len(all_selected_cols) != len(item.data_raw.columns)):
        title = "Clusters for " + ", ".join(all_selected_cols)
    else:
        title = "Clusters for all columns"

    chart3 = figure(title=title, y_axis_label="probability", tools="tap", y_range=y_range, x_range=x_range,
                    width=800)
    chart3.grid.level = "overlay"
    chart3.grid.grid_line_color = "black"
    chart3.grid.grid_line_alpha = 0.05
    chart3.add_layout(Legend(), 'right')
    #chart3.y_range.bounds = (-0.1, 1.1)  # add a little padding around y axis

    # create bands and contours for each group
    legend_items = []
    colors = []
    if add_clusters:
        colors.append([c for c in sorted_data["scatter_group"].unique()]) # add all colors for the different groups
    colors.append("grey") # add grey for the standard group
    colors.append("violet") # add purple for the similar ones
    include_cols = [c for c in all_selected_cols if c != col]
    for i, color in enumerate(colors):
        if (color == 'grey'):
            filtered_data = sorted_data
        elif (color == 'violet'):
            filtered_data = get_similar_items(sorted_data, item, include_cols)
        else:
            filtered_data = sorted_data[sorted_data["scatter_group"] == color].sort_values(by=col)
        if len(filtered_data) > 0:
            if color == 'grey':
                cluster_label = 'standard'
            elif color == 'violet':
                cluster_label = 'similar ' + ", ".join(include_cols[:3]) # only show the first 3 columns to save space TODO improve
            else:
                cluster_label = filtered_data["scatter_label"].iloc[0]
            window = max(len(filtered_data) // 10, 10)
            rolling = filtered_data[item.predict_class].rolling(window=window, center=True, min_periods=1).agg(
                {'lower': lambda ev: ev.quantile(.25, interpolation='lower'),
                 'upper': lambda ev: ev.quantile(.75, interpolation='higher'),
                 'median': 'median',
                 'count': 'count'})
            rolling = rolling.rolling(window=window, center=True, min_periods=1).mean()
            combined = pd.concat([filtered_data, rolling], axis=1)
            combined = ColumnDataSource(combined.reset_index())

            # add legend items
            dummy_for_legend = chart3.line(x=[1, 1], y=[1, 1], line_width=15, color=color, name='dummy_for_legend')
            legend_items.append((cluster_label, [dummy_for_legend]))

            if "contour" in chart_type:
                # only use subset of data for performance reasons
                if len(filtered_data) > 1000:
                    data_subset = filtered_data.sample(n=1000)
                else:
                    data_subset = filtered_data

                x, y, z = kde(data_subset[col], data_subset[item.predict_class], 100)

                # use the color to create a palette
                rgb = color = tuple(int(color[1:][i:i + 2], 16) for i in (0, 2, 4))  # convert hex to rgb
                # to bokeh
                cur_color = bokeh.colors.RGB(*rgb)
                palette = [cur_color]
                for i in range(0, 3):
                    new_color = palette[i].copy()
                    new_color.a -= 0.3
                    palette.append(new_color)

                palette = [c.to_hex() for c in palette]  # convert to hex
                palette = palette[::-1]  # invert the palette

                levels = np.linspace(np.min(z), np.max(z), 7)
                contour = chart3.contour(x, y, z, levels[1:], fill_color=palette, line_color=palette, fill_alpha=0.8)
                contour.fill_renderer.name = cluster_label

                # contour.fill_renderer
                contour_hover = HoverTool(renderers=[contour.fill_renderer], tooltips=[('', '$name')])
                chart3.add_tools(contour_hover)

            if "band" in chart_type:
                band = chart3.varea(x=col, y1='lower', y2='upper', source=combined,
                                    # legend_label=cluster_label,
                                    fill_color=color,
                                    alpha=0.3, name=cluster_label)
                band_hover = HoverTool(renderers=[band], tooltips=[('', '$name')])
                chart3.add_tools(band_hover)

            if "line" in chart_type:
                line = chart3.line(col, 'median', source=combined, color=color, line_width=2,
                                   # legend_label=cluster_label,
                                   name=cluster_label)
                line_hover = HoverTool(renderers=[line], tooltips=[('', '$name')])
                chart3.add_tools(line_hover)

    if "scatter" in chart_type:
        alpha = 0.3
        chart3.scatter(col, item.predict_class, color="scatter_group", source=sorted_data,
                       alpha=alpha, marker='circle', size=3, name="scatter_label",
                       # legend_group="scatter_label"
                       )

    # add the selected item
    if item.type != 'global':
        if (item_style == "point"):
            item_scatter = chart3.scatter(item.data_prob_raw[col], item.data_prob_raw[item.predict_class], color='purple', size=7, name="selected item",
                                          legend_label="selected item")

            scatter_hover = HoverTool(renderers=[item_scatter], tooltips=[('', '$name')])
            chart3.add_tools(scatter_hover)

            # add the point when only selected cols are used
            if item.prob_wo_selected_cols is not None:
                chart3.scatter(x=item.data_prob_raw[col], y=item.prob_wo_selected_cols, color='grey',
                               legend_label='selection probability')

        elif (item_style == "line"):
            item_line = chart3.line(x=[item.data_prob_raw[col], item.data_prob_raw[col]], y=[y_range[0], y_range[1]], line_width=2, color='purple', alpha=0.5,
                                    legend_label="selected item")



    # add legend
    chart3.legend.items.extend([LegendItem(label=x, renderers=y) for (x, y) in legend_items])
    chart3.legend.location = "right"

    # add the "standard probability" line
    chart3.line(x=[x_range[0], x_range[1]], y=[0, 0], line_width=2, color='black', alpha=0.5,
                legend_label='mean probability')


    return chart3
