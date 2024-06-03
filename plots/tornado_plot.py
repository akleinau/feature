from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.models import ColumnDataSource
from plots.render_plot import add_style


def set_col(data, item_source, col):
    if len(item_source.selected.indices) > 0:
        if len(item_source.selected.indices) > 1:
            item_source.selected.indices = item_source.selected.indices[1:2]
        select = data.iloc[item_source.selected.indices]
        select = select['feature'].values[0]
        col[0].value = select  # col[0], bc the widget had to be wrapped in a list to be changed


def shap_tornado_plot(data, col):
    shap = data.shap
    item_source = ColumnDataSource(data=shap)

    plot = figure(title="Feature Set Relevance", y_range=shap['feature_label'], x_range=(-1, 1), tools='tap')
    plot.hbar(
        y='feature_label',
        right='shap_value',
        fill_color=factor_cmap("positive", palette=["steelblue", "crimson"], factors=["pos", "neg"]),
        line_width=0,
        source=item_source,
        nonselection_fill_alpha=0.7,
        selection_hatch_pattern='horizontal_wave',
        selection_hatch_scale=7,
        selection_hatch_weight=1.5,
        selection_hatch_color='purple'
    )

    plot.xaxis.axis_label = "shap value"

    plot = add_style(plot)

    plot.on_event('tap', lambda event: set_col(shap, item_source, col))
    return plot
