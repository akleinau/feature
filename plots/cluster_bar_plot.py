import numpy as np
import pandas as pd
from bokeh.models import HoverTool, FactorRange
from bokeh.plotting import figure
from bokeh.transform import factor_cmap


def cluster_bar_plot(data, item, all_selected_cols, predict_class, predict_label):
    cluster_data = data[[predict_class, "scatter_group", "scatter_label", "group"]].groupby("group")
    clusters = pd.DataFrame()
    clusters['mean'] = cluster_data[predict_class].mean()
    clusters['mean_short'] = clusters['mean'].apply(lambda x: "{:.2f}".format(x) + " ")
    clusters['std'] = cluster_data[predict_class].std()
    clusters['count'] = cluster_data[predict_class].count()
    clusters['median'] = cluster_data[predict_class].median()
    clusters['scatter_label'] = cluster_data['scatter_label'].first()
    clusters['scatter_group'] = cluster_data['scatter_group'].first()
    clusters.reset_index(level=0, inplace=True)
    clusters.sort_values(by='mean', inplace=True)

    y_range = clusters['scatter_label'].values
    x_range = [np.floor(min(clusters['mean'].min(), item.data_prob_raw[predict_class])),
               np.ceil(max(clusters['mean'].max(), item.data_prob_raw[predict_class]))]

    if (len(all_selected_cols) != len(item.data_raw.columns)):
        title = "Clusters for " + ", ".join(all_selected_cols)
    else:
        title = "Clusters for all columns"

    plot = figure(title=title, y_range=y_range, x_range=x_range, width=800)
    plot.hbar(
        y='scatter_label',
        right='mean',
        fill_color="scatter_group",
        line_width=0,
        height=0.5,
        source=clusters,
        nonselection_fill_alpha=0.7,
        border_radius=5,
    )

    #format to two decimal places
    plot.text(x='mean', y='scatter_label', text="mean_short", text_align='right', text_baseline='middle', text_font_size='10pt', text_color="white", source=clusters)

    plot.xaxis.axis_label = predict_label

    # add item
    if item.type != 'global':

        prob_item_label = "selected item (" + "{:.2f}".format(item.data_prob_raw[predict_class]) + ")"
        item_scatter = plot.scatter(y=[item.scatter_label], x=[item.data_prob_raw[predict_class]], color='purple', size=7, name=prob_item_label,
                                      legend_label="selected item")

        scatter_hover = HoverTool(renderers=[item_scatter], tooltips=[('', '$name')])
        plot.add_tools(scatter_hover)



    return plot