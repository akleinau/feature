import panel as pn
from plots.tornado_plot import shap_tornado_plot
from plots.similar_plot import similar_plot
import DataStore as DataStore

pn.extension('floatpanel')

ds = DataStore.DataStore()

# create widgets
ds.get_file_widgets()
ds.get_title_widgets()
ds.get_item_widgets()

# create all the widgets and variables needed for the column group selection
#pn.panel("<br>").servable()
#pn.panel("### Column groups:").servable()
ds.get_row_widgets()

# shap plot
#shap_plot = pn.bind(shap_tornado_plot, ds.param.item, [ds.col])  # col is wrapped to be passed as reference

# remaining layout
pn.Row(pn.bind(lambda a: a.prediction_string(), ds.param.item)).servable()

render_plot = pn.bind(lambda e: e.plot, ds.param.render_plot)
sim_plot = pn.bind(lambda e: e.plot, ds.param.similar_plot)

pn.Row(sim_plot, render_plot, styles=dict(margin='5px 20px 5px 20px')).servable()

ds.get_customization_widgets()

pn.Row(ds.cur_feature, styles=dict(visibility='hidden')).servable() # necessary for the column selection to work