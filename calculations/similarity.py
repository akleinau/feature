import pandas
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from bokeh.palettes import Category20
import numpy as np
from sklearn.tree import _tree


def pred_diff(x1, x2, prediction):
    y1 = x1[prediction].values[0]
    y2 = x2[prediction].values[0]
    diff = y1 - y2
    return diff


def test_setup(data, columns, prediction):
    output = ""
    x1 = data.iloc[[26]]
    x2 = data.iloc[[27]]

    output += "The difference between the two predictions is: " + str(pred_diff(x1, x2, prediction)) + "\n"

    return output


def l2_loss(data, prediction):
    # calculate mean prediction across group
    mean_prediction = data[prediction].mean()
    # calculate difference between each prediction and the mean
    diff = data[prediction] - mean_prediction
    # square the differences
    squared_diff = diff ** 2
    # return the sum of the squared differences
    return squared_diff.sum()


def get_color(x, item_val, range):
    # lambda x: 'saddlebrown' if x >= item_val else 'midnightblue'
    if (x < item_val - range):
        return '#8B4513'
    elif (x > item_val + range):
        return '#191970'
    else:
        return '#800080'


def get_relative_groups(data, col, index):
    item = data.iloc[index]
    labels = {'#228b22': 'All'}

    if len(col) > 1:
        item_val = item[col[1]]
        range = data[col[1]].max() - data[col[1]].min()

        data["scatter_group"] = data[col[1]].apply(lambda x: get_color(x, item_val, range / 20))
        labels['#191970'] = 'higher ' + str(col[1])
        labels['#8B4513'] = 'lower ' + str(col[1])
        labels['#800080'] = 'Similar ' + str(col[1])
    else:
        data["scatter_group"] = '#228b22'

    data["scatter_label"] = data["scatter_group"].apply(lambda x: labels[x])

    return data


def get_tree_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            # add class name at the end
            path += [node]

            paths += [path]

    recurse(0, path, paths)

    rules = {}
    for path in paths:
        rule = ""

        for p in path[:-2]:
            if rule != "":
                rule += " and "
            rule += str(p)
        if rule == "":
            rule = "All"

        # get class name
        class_label = path[-1]
        rules[class_label] = rule

    return rules

def _make_readable_axiom(axiom):
    if (axiom['operator'] == '>' and axiom['value'] == '0.5'):
        return 'higher '
    elif (axiom['operator'] == '<=' and axiom['value'] == '-0.5'):
        return 'lower '
    elif (axiom['operator'] == '>' and axiom['value'] == '-0.5'):
        return 'similar or higher '
    elif (axiom['operator'] == '<=' and axiom['value'] == '0.5'):
        return 'similar or lower '
    else:
        return 'similar '

def make_readable(x):
    axioms = x.split(' and ')
    axioms = [axiom.split() for axiom in axioms]
    if len(axioms[0]) < 3:
        return x

    axioms = pd.DataFrame(axioms, columns=['feature', 'operator', 'value'])
    new_rules = []
    axioms['new_rule'] = axioms.apply(_make_readable_axiom, axis=1)

    #if both'similar/ higher' and 'similar/ lower' exist for a feature, merge them together into 'similar'
    rule_groups = axioms.groupby('feature')
    rule_list = []
    for name, group in rule_groups:
        if 'similar or higher ' in group['new_rule'].values and 'similar or lower ' in group['new_rule'].values:
                rule_list.append('similar ' + group['feature'].values[0])
        else:
            rule_list.extend(group['new_rule'].values + group['feature'].values)

    #now merge the rules together
    return ' and '.join(rule_list)

def shorten_rules(x):
    axioms = x.split(' and ')
    axioms = [axiom[1:-1] for axiom in axioms]
    axioms = [axiom.split() for axiom in axioms]
    if len(axioms[0]) < 3:
        return x


    grouped_axioms = pd.DataFrame(axioms, columns=['feature', 'operator', 'value'])

    #now group
    grouped_axioms= grouped_axioms.groupby(['feature', 'operator'])
    grouped_axioms = grouped_axioms['value'].agg(['max', 'min'])
    grouped_axioms = grouped_axioms.reset_index()
    grouped_axioms['value'] = grouped_axioms.apply(lambda x: x['max'] if x['operator'] == '>' else x['min'], axis=1)

    #now create the new label
    grouped_axioms = grouped_axioms['feature'] + ' ' + grouped_axioms['operator'] + ' ' + grouped_axioms['value']
    return grouped_axioms.str.cat(sep=' and ')


def get_tree_groups(data, all_selected_cols, cur_col, prediction, exclude_col=True):
    # remove the current column from the list of all selected columns
    if exclude_col:
        columns = [col for col in all_selected_cols if col != cur_col]
    else:
        columns = all_selected_cols

    if (len(columns) > 0):
        tree = DecisionTreeRegressor(max_leaf_nodes=4, max_depth=2, min_samples_leaf=0.1)
        tree.fit(data[columns], data[prediction])

        data["group"] = tree.apply(data[columns])

        # create a color for each group
        data["scatter_group"] = data["group"].apply(lambda x: Category20[20][x])

        # create human-readable labels for each group containing the path to the group
        rules = get_tree_rules(tree, columns)

        data["scatter_label"] = data["group"].apply(lambda x: rules[x])
        data["scatter_label"] = data["scatter_label"].apply(shorten_rules)
    else:
        data["scatter_group"] = '#228b22'
        data["scatter_label"] = 'All'

    return data

def get_relative(x, item_val, range):
    # lambda x: 'saddlebrown' if x >= item_val else 'midnightblue'
    if (x < item_val - range):
        return -1
    elif (x > item_val + range):
        return 1
    else:
        return 0

def get_relative_tree_groups(data, all_selected_cols, cur_col, prediction, index, exclude_col=True):
    # remove the current column from the list of all selected columns
    if exclude_col:
        columns = [col for col in all_selected_cols if col != cur_col]
    else:
        columns = all_selected_cols

    if (len(columns) > 0):
        #make new relative columns
        item = data.iloc[index]
        relative_data = pd.DataFrame()
        for col in columns:
            item_val = item[col]
            range = data[col].max() - data[col].min()

            relative_data[col] = data[col].apply(lambda x: get_relative(x, item_val, range / 20))



        tree = DecisionTreeRegressor(max_leaf_nodes=4, max_depth=2, min_samples_leaf=0.1)
        tree.fit(relative_data, data[prediction])

        data["group"] = tree.apply(relative_data)

        # create a color for each group
        data["scatter_group"] = data["group"].apply(lambda x: Category20[20][x])

        # create human-readable labels for each group containing the path to the group
        rules = get_tree_rules(tree, columns)

        data["scatter_label"] = data["group"].apply(lambda x: rules[x])
        data["scatter_label"] = data["scatter_label"].apply(shorten_rules)
        data["scatter_label"] = data["scatter_label"].apply(make_readable)
    else:
        data["scatter_group"] = '#228b22'
        data["scatter_label"] = 'All'

    return data


def get_clustering(cluster_type, data, all_selected_cols, cur_col, prediction, index, exclude_col=True):
    if cluster_type == 'Relative Decision Tree':
        return get_relative_tree_groups(data, all_selected_cols, cur_col, prediction, index, exclude_col)
    else:
        return get_tree_groups(data, all_selected_cols, cur_col, prediction, exclude_col)