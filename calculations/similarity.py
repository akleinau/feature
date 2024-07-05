import pandas as pd

def get_similar_items(data, item, col_white_list):
    use_pdp = False
    if use_pdp:
        return get_pdp_items(data, item, col_white_list)
    else:
        return get_similar_subset(data, item, col_white_list)

def get_similar_subset(data, item, col_white_list):
    use_shap = False
    # standardize the data
    data_std = data.copy()
    item_data = item.data_raw.copy()

    columns = get_columns(col_white_list, data, item_data)

    for col in columns:
        mean = data_std[col].mean()
        std = data_std[col].std()
        data_std[col] = (data_std[col] - mean) / std
        item_data[col] = (item_data[col] - mean) / std

    # calculate distance to item, using shap as weights
    data_std['distance'] = 0
    for col in columns:
        data_std['distance'] += (data_std[col] - item_data[col][0])**2

    # get the 10% closest items
    data_std = data_std.sort_values(by='distance')
    num_items = min(max(int(len(data) * 0.05), 30), len(data_std))
    closest_density = data_std.head(num_items)
    min_distance = len(columns) * 0.1
    closest_distance = data_std[data_std['distance'] <= min_distance]
    combined_indexes = closest_density.index.union(closest_distance.index)

    #map back to original data
    data = data[data.index.isin(combined_indexes)]

    return data


def get_columns(col_white_list, data, item_data):
    if len(col_white_list) == 0:
        columns = list(data.columns)
        excluded_columns = ['prob_', 'scatter', 'prediction', 'group', 'truth']

        columns = [col for col in columns if not any([excluded in col for excluded in excluded_columns])]
        item_columns = [col for col in item_data.columns if not any([excluded in col for excluded in excluded_columns])]
        columns = [col for col in columns if col in item_columns]
    else:
        columns = col_white_list
    return columns


def get_pdp_items(data, item, col_white_list):
    data_pdp = data.copy()
    item_data = item.data_raw.copy()
    columns = get_columns(col_white_list, data, item_data)

    # replace each column with the item value
    for col in columns:
        data_pdp[col] = item_data[col].values[0]

    return data_pdp

