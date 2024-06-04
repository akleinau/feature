from plots.cluster_bar_plot import cluster_bar_plot
from plots.dependency_plot import dependency_scatterplot
from plots.parallel_plot import parallel_plot
import panel as pn


class RenderPlot:

    def __init__(self, graph_type, all_selected_cols, clustered_data, cur_feature, item, item_index, chart_type, predict_class):
        self.plot = self.get_render_plot(graph_type, all_selected_cols, clustered_data, cur_feature, item, item_index,
                                         chart_type, predict_class)

    def get_render_plot(self, graph_type, all_selected_cols, clustered_data, cur_feature, item, item_index,
                        chart_type, predict_class):
        if graph_type == 'Cluster':
            plot = cluster_bar_plot(clustered_data, item, all_selected_cols, predict_class)
            plot = add_style(plot)
            return plot
        elif graph_type == 'Dependency':
            dep_plot = dependency_scatterplot(clustered_data, cur_feature.value, all_selected_cols,
                                              item, chart_type)
            dep_plot = add_style(dep_plot)
            return pn.Column(dep_plot, cur_feature)
        else:
            plot = parallel_plot(clustered_data, cur_feature.value, all_selected_cols,
                                 item.prediction, item.data, chart_type)
            plot = add_style(plot)
            return plot

def add_style(plot):
    plot.title.text_font_size = '20px'
    plot.title.align = 'center'
    plot.xaxis.major_label_text_font_size = '14px'
    plot.yaxis.major_label_text_font_size = '14px'
    plot.xaxis.axis_label_text_font_size = '14px'
    if len(plot.legend) >0:
        plot.legend.label_text_font_size = '14px'
    if len(plot.hover) > 0:
        # set the font size of the hover tooltip. I don't think this actually works?
        for h in plot.hover:
            h.tooltips.text_font_size = '14px'

    plot.grid.grid_line_color = "black"
    plot.grid.grid_line_alpha = 0.0

    return plot