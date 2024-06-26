import param
import panel as pn
import calculations.data_loader as data_loader
import calculations.item_functions as item_functions
import calculations.column_functions as column_functions
import calculations.clusters as clusters
import plots.render_plot as render_plot
import plots.similar_plot as similar_plot


class DataStore(param.Parameterized):
    item = param.ClassSelector(class_=item_functions.Item)
    columnGrouping = param.ClassSelector(class_=column_functions.ColumnGrouping)
    data_loader = param.ClassSelector(class_=data_loader.DataLoader)
    all_selected_cols = param.List()
    clustering = param.ClassSelector(class_=clusters.Clustering)
    render_plot = param.ClassSelector(class_=render_plot.RenderPlot)
    similar_plot = param.ClassSelector(class_=similar_plot.SimilarPlot)

    def __init__(self, **params):
        super().__init__(**params)
        self.active = True
        self.file = pn.widgets.FileInput(accept='.csv', name='Upload data')
        self.nn_file = pn.widgets.FileInput(accept='.pkl', name='Upload neural network')
        self.truth_file = pn.widgets.FileInput(accept='.csv', name='Upload truth')
        self.calculate = pn.widgets.Button(name='Calculate', button_type='primary', styles=dict(margin='auto'))
        self.calculate.on_click(self.update_data)
        self.data_loader = data_loader.DataLoader()

        # item
        self.item_type = pn.widgets.RadioButtonGroup(name='item type', options=['predefined', 'custom', 'global'], value='predefined')
        self.item_index = pn.widgets.EditableIntSlider(name='item index', start=0, end=100, value=26, width=250)
        self.item_custom_content = pn.Column()

        # predict class
        self.predict_class = pn.widgets.Select(name='prediction', options=list(self.data_loader.classes), width=250)
        self.predict_class_label = pn.widgets.TextInput(name='prediction label', value=self.predict_class.value, width=250)
        self.predict_class.param.watch(lambda event: self.predict_class_label.param.update(value=event.new),
                                        parameter_names=['value'], onlychanged=False)

        # columns
        self.col_type = 'singular'
        self.col = pn.widgets.Select(name='column', options=self.data_loader.columns)
        self.all_selected_cols = []
        self.all_selected_cols_widget = pn.widgets.MultiChoice(name='Interacting Features', options=self.data_loader.columns, value=self.all_selected_cols)
        self.col.param.watch(
            lambda event: self.param.update(all_selected_cols=column_functions.return_col(event.new)),
            parameter_names=['value'], onlychanged=False)
        self.all_selected_cols_widget.param.watch(
            lambda event: self.param.update(all_selected_cols=event.new),
            parameter_names=['value'], onlychanged=False)

        # groups
        self.cur_feature = pn.widgets.Select(name='', options=self.all_selected_cols,
                                             value=None, align='center')
        self.param.watch(lambda event: self.cur_feature.param.update(options=event.new, value = self.get_first(event.new)),
                         parameter_names=['all_selected_cols'], onlychanged=False)
        #self.all_selected_cols_widget.param.watch(lambda event: self.cur_feature.param.update(options=self.get_all_selected_cols(event.new), value=self.get_all_selected_cols(event.new)[0]),
         #                                           parameter_names=['value'], onlychanged=False)
        self.column_grouping = column_functions.ColumnGrouping(self.data_loader.columns)
        self.column_grouping.param.watch(self.column_grouping_changed, parameter_names=['combined_columns'],
                                         onlychanged=False)

        # customization widgets
        self.cluster_type = pn.widgets.Select(name='cluster_type', options=['Relative Decision Tree', 'Decision Tree',
                                                                            'Similarity Decision Tree',
                                                                            'SimGroup Decision Tree'],
                                              value='Decision Tree')
        self.chart_type = pn.widgets.MultiChoice(name='chart_type', options=['scatter', 'line', 'band', 'contour'],
                                                 value=['line'])

        self.graph_type = pn.widgets.Select(name='graph_type', options=['Cluster', 'Dependency', 'Parallel'],
                                            value='Cluster')
        self.num_leafs = pn.widgets.EditableIntSlider(name='num_leafs', start=1, end=15, value=3)

        # item
        self.item = self._update_item_self()
        self.item_index.param.watch(self.update_item_self, parameter_names=['value'],
                                    onlychanged=False)
        self.predict_class_label.param.watch(self.update_item_self, parameter_names=['value'],
                                       onlychanged=False)
        self.item_type.param.watch(self.update_item_self, parameter_names=['value'],
                                    onlychanged=False)
        self.init_item_custom_content()

        # clustered data
        self.clustering = self._update_clustered_data()
        self.cur_feature.param.watch(self.update_clustered_data, parameter_names=['value'], onlychanged=False)
        self.item_index.param.watch(self.update_clustered_data, parameter_names=['value'], onlychanged=False)
        self.item_type.param.watch(self.update_clustered_data, parameter_names=['value'], onlychanged=False)
        self.cluster_type.param.watch(self.update_clustered_data, parameter_names=['value'], onlychanged=False)
        self.num_leafs.param.watch(self.update_clustered_data, parameter_names=['value'], onlychanged=False)
        self.predict_class.param.watch(self.update_clustered_data, parameter_names=['value'], onlychanged=False)

        # render
        self.render_plot = self._update_render_plot()
        self.param.watch(self.update_render_plot,
            parameter_names=['clustering'], onlychanged=False)
        self.graph_type.param.watch(self.update_render_plot,
                                    parameter_names=['value'], onlychanged=False)
        self.chart_type.param.watch(lambda event: self.update_render_plot(event, caused_by_chart=True),
                                    parameter_names=['value'], onlychanged=False)
        self.predict_class_label.param.watch(self.update_render_plot, parameter_names=['value'],
                                       onlychanged=False)

        # render similar plot
        self.similar_plot = self._update_similar_plot()
        self.cur_feature.param.watch(self.update_similar_plot,
                                parameter_names=['value'], onlychanged=False)
        self.item_index.param.watch(self.update_similar_plot,
                                parameter_names=['value'], onlychanged=False)
        self.item_type.param.watch(self.update_similar_plot,
                                parameter_names=['value'], onlychanged=False)

    def prediction_string(self):
        return pn.bind(lambda x: x.item.prediction_string(), self)

    def column_grouping_changed(self, event):
        self.update_item_self()

    def update_data(self, event):
        self.active = False
        loader = data_loader.DataLoader(self.file.value, self.nn_file.value, self.truth_file.value)
        predict_class = loader.classes[0]
        all_selected_cols = []
        self.all_selected_cols_widget.options = loader.columns
        self.all_selected_cols_widget.value = []
        cur_feature = None
        cur_feature_widget = pn.widgets.Select(name='', options=all_selected_cols,
                                             value=cur_feature, align='center')
        item = item_functions.Item(loader, loader.data_and_probabilities, "predefined", self.item_index.value, None, predict_class, predict_class)
        clustering = clusters.Clustering(self.cluster_type.value, loader.data_and_probabilities, all_selected_cols,
                                           cur_feature, predict_class, item, num_leafs=self.num_leafs.value,
                                           exclude_col=False)

        self.predict_class.param.update(options=loader.classes, value=predict_class)

        self.param.update(data_loader=loader, item=item, clustering=clustering, all_selected_cols=all_selected_cols,
                          render_plot=render_plot.RenderPlot(self.graph_type.value, all_selected_cols,
                                                             clustering.data, cur_feature_widget, item,
                                                             self.item_index.value, self.chart_type, predict_class, predict_class, active_tab=1))

        self.col.param.update(options=loader.columns)

        self.column_grouping.init_groups(loader.columns)

        self.cur_feature.param.update(options=self.all_selected_cols, value=cur_feature)

        self.init_item_custom_content()

        self.render_plot = self._update_render_plot()
        self.similar_plot = self._update_similar_plot()

        self.active = True


    def init_item_custom_content(self):
        self.item_custom_content.clear()
        self.item_custom_content.append("(missing values will be imputed)")
        for col in self.data_loader.columns:
            self.item_custom_content.append(pn.widgets.LiteralInput(name=col, value=None))
        button = pn.widgets.Button(name="Calculate", button_type="primary", align="center", width=300)
        self.item_custom_content.append(button)
        button.on_click(self.update_item_self)

    def _update_clustered_data(self):
        return clusters.Clustering(self.cluster_type.value, self.data_loader.data_and_probabilities,
                                     self.all_selected_cols,
                                     self.cur_feature.value, self.predict_class.value, self.item,
                                     exclude_col=False, num_leafs=self.num_leafs.value)

    def update_clustered_data(self, *params):
        if self.active:
            self.param.update(
                clustering=self._update_clustered_data())

    def get_first(self, event):
        if len(event) > 0:
            return event[0]
        return None

    def get_all_data(self):
        return pn.bind(data_loader.load_data, self.file.value, self.data_loader.nn)

    def get_file_widgets(self):
        return pn.Column(pn.Row(
            pn.Column("Data*:", "Model*:", "Truth:"),
            pn.Column(self.file, self.nn_file,  self.truth_file)),
            self.calculate,
            styles=dict(padding_bottom='10px', margin='0', align='end'))

    def get_title_widgets(self):
        return pn.Column(self.predict_class, self.predict_class_label, styles=dict(padding_top='10px'))

    def get_item_widgets(self):
        second_item = pn.bind(lambda t: self.item_index if t == 'predefined' else self.item_custom_content if t == 'custom' else None, self.item_type)
        return pn.Column(self.item_type, second_item)

    def get_customization_widgets(self):
        return pn.Row(self.cluster_type, self.num_leafs)

    def get_row_widgets(self):
        if (self.col_type == 'singular'):
            return pn.Row(self.all_selected_cols_widget, styles=dict(margin="auto"))
        return self.column_grouping.row

    def _update_render_plot(self, caused_by_chart=False):
        active_tab = 1 if caused_by_chart else 1
        return render_plot.RenderPlot(self.graph_type.value, self.all_selected_cols,
                                      self.clustering.data, self.cur_feature, self.item,
                                      self.item_index.value,
                                      self.chart_type, self.predict_class.value, self.predict_class_label.value,
                                      active_tab)
    def update_render_plot(self, event, caused_by_chart=False):
        if self.active:
            self.param.update(render_plot=self._update_render_plot(caused_by_chart))

    def _update_similar_plot(self):
        return similar_plot.SimilarPlot(self.data_loader, self.item, self.all_selected_cols, self.cur_feature.value)

    def update_similar_plot(self, *params):
        if self.active:
            self.param.update(similar_plot= self._update_similar_plot())

    def _update_item_self(self):
            return item_functions.Item(self.data_loader, self.data_loader.data_and_probabilities, self.item_type.value,
                                         self.item_index.value, self.item_custom_content,
                                         self.predict_class.value, self.predict_class_label.value,
                                         self.column_grouping.combined_columns)

    def update_item_self(self, *params):
        if self.active:
            self.param.update(item=self._update_item_self())