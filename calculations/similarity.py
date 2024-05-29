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
        labels['#191970'] = 'Higher ' + str(col[1])
        labels['#8B4513'] = 'Lower ' + str(col[1])
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
        # get class name
        class_label = path[-1]
        rules[class_label] = rule

    return rules


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
    else:
        data["scatter_group"] = '#228b22'
        data["scatter_label"] = 'All'

    return data


def get_clustering(cluster_type, data, all_selected_cols, cur_col, prediction, index, exclude_col=True):
    if cluster_type == 'Relative':
        return get_relative_groups(data, all_selected_cols, index)
    else:
        return get_tree_groups(data, all_selected_cols, cur_col, prediction, exclude_col)
