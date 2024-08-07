import pandas as pd
import panel as pn
import param
from bokeh.models import BoxAnnotation
from bokeh.plotting import figure
from panel.viewable import Viewer, Viewable

from calculations.feature_iter import FeatureIter
from calculations.similarity import get_window_items
from plots.dependency_plot import get_rolling
from plots.styling import add_style, style_options
import plots.dependency_plot as dependency_plot


class OverviewPlot(Viewer):
    ranked_plots = param.ClassSelector(class_=pn.FlexBox)
    all_selected_cols = param.List()

    def __init__(self):
        super().__init__()

        self.ranked_plots = pn.FlexBox()
        self.dps = []
        self.toggle_widget = pn.widgets.RadioButtonGroup(options=['change in prediction',
                                                                  'interaction effect'],
                                                         value='change in prediction',
                                                         button_style='outline', stylesheets=[style_options])
        self.toggle_widget.param.watch(self.toggle_changed, parameter_names=['value'], onlychanged=False)

        self.all_selected_cols = []

    @param.depends('ranked_plots')
    @param.depends('all_selected_cols')
    def __panel__(self) -> Viewable:
        if len(self.ranked_plots.objects) == 0:
            return pn.Column()
        elif len(self.all_selected_cols) == 0:
            return pn.Column("## Feature Overview: ",
                             pn.Row(
                                 pn.pane.Markdown("**mean prediction per feature value**", styles=dict(color='#606060')),
                                 pn.pane.Markdown("**selected item**", styles=dict(color='#19b57A')),
                                 # pn.pane.Markdown("**mean prediction**", styles=dict(color='#A0A0A0')),
                                 # pn.pane.Markdown("**positive influence**", styles=dict(color='#AE0139')),
                                 # pn.pane.Markdown("**negative influence**", styles=dict(color='#3801AC')),
                             ),
                             self.ranked_plots)
        else:
            return pn.Column()

    def add_feature_view(self):
        return pn.Column(
            "(ranked by strongest interaction effect at item value)",
            self.toggle_widget,
            self.ranked_plots,
        )

    def update(self, data, item, y_col, feature_iter, recommendation, data_loader):
        """
        updates the plot with the new data

        :param data: pd.DataFrame
        :param item: Item
        :param y_col: str
        :param columns: list
        :param all_selected_cols: list
        :param feature_iter: FeatureIter
        :return:
        """
        self.all_selected_cols = feature_iter.all_selected_cols
        self.toggle_widget.value = "change in prediction"

        ranked_plots = pn.FlexBox()
        dps = []

        # go through each row of the dataset recommendation.dataset_single
        for index, row in recommendation.dataset_single.iterrows():

            col = row['feature']
            dp = dependency_plot.DependencyPlot(simple=True)
            dp.update_plot(data, self.all_selected_cols + [col], item, data_loader, feature_iter, True, False)
            #dp.toggle_widget.value = "interaction effect"

            dps.append(dp)
            ranked_plots.append(dp.plot)

        #reverse the order
        ranked_plots.objects = ranked_plots.objects[::-1]

        self.ranked_plots = ranked_plots
        self.dps = dps

    def toggle_changed(self, event):

        ranked_plots = pn.FlexBox()
        for dp in self.dps:
            dp.toggle_widget.value = event.new
            ranked_plots.append(dp.plot)

        ranked_plots.objects = ranked_plots.objects[::-1]
        self.ranked_plots = ranked_plots

    def hide_all(self):
        self.all_selected_cols = []
        self.ranked_plots = pn.FlexBox()
        self.dps = []
