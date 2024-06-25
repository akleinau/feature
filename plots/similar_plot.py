from bokeh.plotting import figure
from bokeh.transform import jitter
from bokeh.layouts import column, layout
from calculations.similarity import get_similar_items
import param

class SimilarPlot(param.Parameterized):

    def __init__(self, data_loader, item, all_selected_cols, cur_feature=None, **params):
        super().__init__(**params)
        self.plot = similar_plot(data_loader, item, all_selected_cols, cur_feature)

def similar_plot(data_loader, item, all_selected_cols, cur_feature):
    column_criteria = "curr"

    include_cols = []
    if column_criteria == "selected":
        include_cols = all_selected_cols
    elif column_criteria == "curr":
        include_cols = [col for col in all_selected_cols if col != cur_feature]

    data = data_loader.data.copy()
    data['fixed'] = 1
    similar_item_group = get_similar_items(data, item, include_cols)

    # normalize the data
    normalized_data = data.copy()
    normalized_similar_item_group = similar_item_group.copy()
    for col in data.columns.drop('fixed'):
        mean = data[col].mean()
        std = data[col].std()
        normalized_data[col] = (data[col] - mean) / std
        normalized_similar_item_group[col] = (similar_item_group[col] - mean) / std

    # calculate the difference per column
    data_mean = normalized_data.mean()
    similar_item_group_mean = normalized_similar_item_group.mean()
    diff = data_mean - similar_item_group_mean
    diff = diff.drop('fixed')
    diff = diff.abs()
    diff = diff.sort_values(ascending=False)

    # for each column, create a bokeh plot with the distribution of the data
    plot_list = []

    display_cols = diff.index.tolist()
    include_cols_for_display = all_selected_cols if len(all_selected_cols) > 0 else data_loader.data.columns
    display_cols = [col for col in display_cols if col in include_cols_for_display]

    for col in display_cols:

        # create a figure
        x_range = [data[col].min(), data[col].max()]
        plot = figure(title="Similar items", x_range=x_range, toolbar_location=None, height=80, width=400, sizing_mode='fixed')

        # add points
        #plot.scatter(x=jitter(col, 3), y=jitter('fixed', 2), alpha=0.1, source=data, size=2, color='grey')
        plot.scatter(x=jitter(col, 0.5), y=jitter('fixed', 2), alpha=0.2, source=similar_item_group, size=5, color='purple')


        # add the mean of the data and of similar_item_group as lines
        data_mean = data[col].mean()
        similar_item_group_mean = similar_item_group[col].mean()
        plot.line([data_mean, data_mean], [0, 2], color='grey', line_width=2, alpha=0.9)
        #plot.line([similar_item_group_mean, similar_item_group_mean], [0, 2], color='#9932CC', line_width=2)

        # add item as a red dot
        plot.scatter(x=item.data_raw[col], y=1, size=7, color='lightseagreen')

        plot.yaxis.axis_label = col + " = " + "{:.2f}".format(item.data_raw[col].values[0])
        plot.yaxis.axis_label_orientation = "horizontal"
        #hide ticks of the yaxis but not the label
        plot.yaxis.major_tick_line_color = None
        plot.yaxis.minor_tick_line_color = None
        plot.yaxis.major_label_text_font_size = '0pt'
        #hide grid
        plot.ygrid.grid_line_color = None
        plot.xgrid.grid_line_color = None


        plot.title.visible = False
        plot_list.append(plot)


    return layout(plot_list)


