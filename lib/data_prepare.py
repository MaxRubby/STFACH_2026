import datetime
import pandas as pd
import torch
import numpy as np
import os
from tslearn.metrics import cdist_dtw
from tqdm import tqdm

import controldiffeq
from .utils import print_log, StandardScaler 

def make_dir(name):
    if (os.path.exists(name)):
        print('has  save path')
    else:
        os.makedirs(name)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def process_data(data, times):
    """
    Process data for CDE (Controlled Differential Equations).
    After add_window_horizon, data shape is:
    - 3D point data: (samples, timesteps, nodes, features)
    - 4D grid data: (samples, timesteps, row, col, features) or (samples, timesteps, nodes, features)
    """
    augmented_data = []
    # data.shape: (samples, timesteps, nodes/spatial, features)
    # times: (timesteps,)
    # Repeat times for all samples and spatial dimensions
    if len(data.shape) == 4:
        # (samples, timesteps, nodes, features)
        times_repeated = times.unsqueeze(0).unsqueeze(0).repeat(data.shape[0], data.shape[2], 1).unsqueeze(-1).transpose(1, 2)
    elif len(data.shape) == 5:
        # (samples, timesteps, row, col, features) - need to flatten spatial dims
        batch_size, timesteps, row, col, features = data.shape
        times_repeated = times.unsqueeze(0).unsqueeze(0).repeat(batch_size, row * col, 1).unsqueeze(-1).transpose(1, 2)
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}. Expected 4D or 5D after add_window_horizon.")
    
    augmented_data.append(times_repeated)
    augmented_data.append(torch.tensor(data[..., :]))
    return torch.cat(augmented_data, dim=3)

def calculate_adjacency_matrix(adj_mx,weight_adj_epsilon=0.1):
    print("Start Calculate the weight by Gauss kernel!")
    distances = adj_mx[~np.isinf(adj_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(adj_mx / std))
    adj_mx[adj_mx < weight_adj_epsilon] = 0
    return adj_mx

def add_external_information_grid(df, timesolts, add_time_in_day, add_day_in_week, idx_of_ext_timesolts=None, ext_data=None, use_row_column=True):
    """Add external information for grid data"""
    is_time_nan = np.isnan(timesolts).any()
    data_list = [df]
    
    if use_row_column:
        # 4D data: (time, row, column, feature)
        num_samples, len_row, len_column, feature_dim = df.shape
        print(df.shape)
        
        if add_time_in_day and not is_time_nan:
            time_ind = (timesolts - timesolts.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, len_row, len_column, 1]).transpose((3, 1, 2, 0))
            data_list.append(time_in_day)
        if add_day_in_week and not is_time_nan:
            dayofweek = []
            for day in timesolts.astype("datetime64[D]"):
                dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
            day_in_week = np.zeros(shape=(num_samples, len_row, len_column, 7))
            day_in_week[np.arange(num_samples), :, :, dayofweek] = 1
            day_in_week2 = np.argmax(day_in_week, axis=3).astype(np.float32)[..., np.newaxis]
            data_list.append(day_in_week2)
        if ext_data is not None:
            if not is_time_nan:
                indexs = []
                for ts in timesolts:
                    ts_index = idx_of_ext_timesolts[ts]
                    indexs.append(ts_index)
                select_data = ext_data[indexs]
                for i in range(select_data.shape[1]):
                    data_ind = select_data[:, i]
                    data_ind = np.tile(data_ind, [1, len_row, len_column, 1]).transpose((3, 1, 2, 0))
                    data_list.append(data_ind)
            else:
                if ext_data.shape[0] == df.shape[0]:
                    select_data = ext_data
                    for i in range(select_data.shape[1]):
                        data_ind = select_data[:, i]
                        data_ind = np.tile(data_ind, [1, len_row, len_column, 1]).transpose((3, 1, 2, 0))
                        data_list.append(data_ind)
    else:
        # 3D data: (time, nodes, feature)
        num_samples, num_nodes, feature_dim = df.shape
        print(df.shape)
        
        if add_time_in_day and not is_time_nan:
            time_ind = (timesolts - timesolts.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
        if add_day_in_week and not is_time_nan:
            dayofweek = []
            for day in timesolts.astype("datetime64[D]"):
                dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
            day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
            day_in_week[np.arange(num_samples), :, dayofweek] = 1
            day_in_week2 = np.argmax(day_in_week, axis=2).astype(np.float32)[..., np.newaxis]
            data_list.append(day_in_week2)
        if ext_data is not None:
            if not is_time_nan:
                indexs = []
                for ts in timesolts:
                    ts_index = idx_of_ext_timesolts[ts]
                    indexs.append(ts_index)
                select_data = ext_data[indexs]
                for i in range(select_data.shape[1]):
                    data_ind = select_data[:, i]
                    data_ind = np.tile(data_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
                    data_list.append(data_ind)
            else:
                if ext_data.shape[0] == df.shape[0]:
                    select_data = ext_data
                    for i in range(select_data.shape[1]):
                        data_ind = select_data[:, i]
                        data_ind = np.tile(data_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
                        data_list.append(data_ind)
    
    data = np.concatenate(data_list, axis=-1)
    return data

def add_external_information(df, timesolts,add_time_in_day, add_day_in_week,idx_of_ext_timesolts=None,ext_data=None):
    num_samples, num_nodes, feature_dim = df.shape
    is_time_nan = np.isnan(timesolts).any()
    data_list = [df]
    print(df.shape)

    if add_time_in_day and not is_time_nan:
        time_ind = (timesolts - timesolts.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week and not is_time_nan:
        dayofweek = []
        for day in timesolts.astype("datetime64[D]"):
            dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, dayofweek] = 1

        day_in_week2 = np.argmax(day_in_week, axis=2).astype(np.float32)[...,np.newaxis]
        data_list.append(day_in_week2)
    if ext_data is not None:
        if not is_time_nan:
            indexs = []
            for ts in timesolts:
                ts_index = idx_of_ext_timesolts[ts]
                indexs.append(ts_index)
            select_data = ext_data[indexs]
            for i in range(select_data.shape[1]):
                data_ind = select_data[:, i]
                data_ind = np.tile(data_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
                data_list.append(data_ind)
        else:
            if ext_data.shape[0] == df.shape[0]:
                select_data = ext_data
                for i in range(select_data.shape[1]):
                    data_ind = select_data[:, i]
                    data_ind = np.tile(data_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
                    data_list.append(data_ind)
    data = np.concatenate(data_list, axis=-1)
    return data

def load_grid_geo(data_path, filename):
    """Load grid geo file and return geo information"""
    geofile = pd.read_csv(data_path + filename + '.geo')
    geo_ids = list(geofile['geo_id'])
    num_nodes = len(geo_ids)
    geo_to_ind = {}
    geo_to_rc = {}
    for index, idx in enumerate(geo_ids):
        geo_to_ind[idx] = index
    for i in range(geofile.shape[0]):
        geo_to_rc[geofile['geo_id'][i]] = [geofile['row_id'][i], geofile['column_id'][i]]
    len_row = max(list(geofile['row_id'])) + 1
    len_column = max(list(geofile['column_id'])) + 1
    print("Loaded file " + filename + '.geo' + ', num_grids=' + str(len(geo_ids))
          + ', grid_size=' + str((len_row, len_column)))
    return geo_ids, num_nodes, geo_to_ind, geo_to_rc, len_row, len_column

def load_grid_rel(data_path, filename, num_nodes, len_row, len_column, weight_adj_epsilon=0.1):
    """Load or generate grid adjacency matrix"""
    rel_path = data_path + filename + '.rel'
    if os.path.exists(rel_path):
        return load_rel(data_path, filename, weight_adj_epsilon=weight_adj_epsilon)
    else:
        adj_mx = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        dirs = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        for i in range(len_row):
            for j in range(len_column):
                index = i * len_column + j
                for d in dirs:
                    nei_i = i + d[0]
                    nei_j = j + d[1]
                    if nei_i >= 0 and nei_i < len_row and nei_j >= 0 and nei_j < len_column:
                        nei_index = nei_i * len_column + nei_j
                        adj_mx[index][nei_index] = 1
                        adj_mx[nei_index][index] = 1
        print("Generated grid rel file, shape=" + str(adj_mx.shape))
        return adj_mx, num_nodes

def load_rel(data_path , filename,set_weight_link_or_dist='dist',init_weight_inf_or_zero='zero',bidir=True,calculate_weight_adj=False,weight_adj_epsilon=0.1):
 
    relfile = pd.read_csv(data_path+filename + '.rel')
    geofile = pd.read_csv(data_path + filename + '.geo')
    geo_ids = list(geofile['geo_id'])
    num_nodes = len(geo_ids)
    geo_to_ind = {}
    for index, idx in enumerate(geo_ids):
        geo_to_ind[idx] = index

    print('set_weight_link_or_dist: {}'.format(set_weight_link_or_dist))
    print('init_weight_inf_or_zero: {}'.format(init_weight_inf_or_zero))
    weight_col = ''
    if weight_col != '':
        if isinstance(weight_col, list):
            if len(weight_col) != 1:
                raise ValueError('`weight_col` parameter must be only one column!')
            weight_col = weight_col[0]
        distance_df = relfile[~relfile[weight_col].isna()][[
            'origin_id', 'destination_id', weight_col]]
    else:
        if len(relfile.columns) > 5:
            print_log("warning!!!!!!!!!rel file column number != 5 columns! default to fifth column as weight_col")
            weight_col = relfile.columns[4]
            distance_df = relfile[~relfile[weight_col].isna()][[
                'origin_id', 'destination_id', weight_col]]
        else:
            weight_col = relfile.columns[-1]
            distance_df = relfile[~relfile[weight_col].isna()][[
                'origin_id', 'destination_id', weight_col]]
    adj_mx = np.zeros((len(geo_ids), len(geo_ids)), dtype=np.float32)
    if init_weight_inf_or_zero.lower() == 'inf' and set_weight_link_or_dist.lower() != 'link':
        adj_mx[:] = np.inf
    for row in distance_df.values:
        if row[0] not in geo_to_ind or row[1] not in geo_to_ind:
            continue
        if set_weight_link_or_dist.lower() == 'dist':
            adj_mx[geo_to_ind[row[0]], geo_to_ind[row[1]]] = row[2]
            if bidir:
                adj_mx[geo_to_ind[row[1]], geo_to_ind[row[0]]] = row[2]
        else:
            adj_mx[geo_to_ind[row[0]], geo_to_ind[row[1]]] = 1
            if bidir:
                adj_mx[geo_to_ind[row[1]], geo_to_ind[row[0]]] = 1
    print("Loaded file " + filename + '.rel &.geo,adj_mx shape=' + str(adj_mx.shape))
    if calculate_weight_adj:
        adj_mx = calculate_adjacency_matrix(adj_mx, weight_adj_epsilon=weight_adj_epsilon)
    return adj_mx, num_nodes

def load_grid(data_path, filename, geo_ids, len_row, len_column, data_col, load_external=True, use_row_column=True):
    """Load grid data file (.grid)"""
    print_log("Loading file " + filename + '.grid')
    gridfile = pd.read_csv(data_path + filename + '.grid')
    if data_col != '':
        if isinstance(data_col, list):
            data_col_copy = data_col.copy()
        else:
            data_col_copy = [data_col].copy()
        data_col_copy.insert(0, 'time')
        data_col_copy.insert(1, 'row_id')
        data_col_copy.insert(2, 'column_id')
        gridfile = gridfile[data_col_copy]
    else:
        gridfile = gridfile[gridfile.columns[2:]]
    
    timesolts = list(gridfile['time'][:int(gridfile.shape[0] / len(geo_ids))])
    idx_of_timesolts = dict()
    if not gridfile['time'].isna().any():
        timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), timesolts))
        timesolts = np.array(timesolts, dtype='datetime64[ns]')
        for idx, _ts in enumerate(timesolts):
            idx_of_timesolts[_ts] = idx
    
    feature_dim = len(gridfile.columns) - 3
    df = gridfile[gridfile.columns[-feature_dim:]]
    len_time = len(timesolts)
    
    if use_row_column:
        # Load as 4D: (time, row, column, feature)
        data = []
        for i in range(len_row):
            tmp = []
            for j in range(len_column):
                index = (i * len_column + j) * len_time
                tmp.append(df[index:index + len_time].values)
            data.append(tmp)
        data = np.array(data, dtype=np.float64)
        data = data.swapaxes(2, 0).swapaxes(1, 2)
    else:
        # Load as 3D: (time, nodes, feature)
        data = []
        for i in range(0, df.shape[0], len_time):
            data.append(df[i:i + len_time].values)
        data = np.array(data, dtype=np.float64)
        data = data.swapaxes(0, 1)
    
    print_log("Loaded file " + filename + '.grid' + ', shape=' + str(data.shape))
    
    if load_external:
        data = add_external_information_grid(data, timesolts=timesolts, idx_of_ext_timesolts=None, 
                                             add_time_in_day=True, add_day_in_week=True, 
                                             ext_data=None, use_row_column=use_row_column)
    return data.astype(np.float32), timesolts, idx_of_timesolts

def load_dyna(data_path, filename, geo_ids, data_col,load_external=True):
    print_log("Loading file " + filename + '.dyna')
    dynafile = pd.read_csv(data_path + filename + '.dyna')
    if data_col != '':
        if isinstance(data_col, list):
            data_col = data_col.copy()
        else:
            data_col = [data_col].copy()
        data_col.insert(0, 'time')
        data_col.insert(1, 'entity_id')
        dynafile = dynafile[data_col]
    else:
        dynafile = dynafile[dynafile.columns[2:]]
    timesolts = list(dynafile['time'][:int(dynafile.shape[0] / len(geo_ids))])
    idx_of_timesolts = dict()
    if not dynafile['time'].isna().any():
        timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), timesolts))
        timesolts = np.array(timesolts, dtype='datetime64[ns]')
        for idx, _ts in enumerate(timesolts):
            idx_of_timesolts[_ts] = idx
    feature_dim = len(dynafile.columns) - 2
    df = dynafile[dynafile.columns[-feature_dim:]]
    len_time = len(timesolts)
    data = []
    for i in range(0, df.shape[0], len_time):
        data.append(df[i:i+len_time].values)
    data = np.array(data, dtype=np.float32)
    data = data.swapaxes(0, 1)
    print_log("Loaded file " + filename + '.dyna' + ', shape=' + str(data.shape))
    if load_external:
        data = add_external_information(data,timesolts = timesolts, idx_of_ext_timesolts=None, add_time_in_day=True, add_day_in_week=True, ext_data = None)
    return data.astype(np.float32)


def add_window_horizon(df, input_window = 12, output_window = 12):
    num_samples = df.shape[0]
    x_offsets = np.sort(np.concatenate((np.arange(-input_window + 1, 1, 1),)))
    y_offsets = np.sort(np.arange(1, output_window + 1, 1))

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x_t = df[t + x_offsets, ...]
        y_t = df[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y
 
def split_train_val_test( x, y, train_rate = 0.6, eval_rate = 0.2, test_rate = 0.2):
    test_rate = 1 - train_rate - eval_rate
    num_samples = x.shape[0]
    num_test = round(num_samples * test_rate)
    num_train = round(num_samples * train_rate)
    num_val = num_samples - num_test - num_train

    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
    x_test, y_test = x[-num_test:], y[-num_test:]
    print_log("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
    print_log("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
    print_log("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

 
 


def shift_traffic_data(data, num, fill_value=0):
    """
    Shifts array along the first axis by the number of steps given by num.
    Positive num shifts down, negative num shifts up.
    fill_value is used to replace the missing values after the shift.
    After add_window_horizon, supports:
    - 4D data: (samples, timesteps, nodes, features)
    - 5D data: (samples, timesteps, row, column, features)
    """
    if len(data.shape) == 4:
        # 4D data after horizon: (samples, timesteps, nodes, features)
        arr = data[:, :, :, :1]
    elif len(data.shape) == 5:
        # 5D data after horizon: (samples, timesteps, row, column, features)
        arr = data[:, :, :, :, :1]
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}. Expected 4D or 5D after add_window_horizon.")
    
    result = np.full_like(arr, fill_value=fill_value, dtype=np.float64)
    if num > 0:  # Shift down
        result[:num] = arr[:num]
        result[num:] = arr[:-num]
    elif num < 0:  # Shift up
        result[num:] = arr[num:]
        result[:num] = arr[-num:]
    else:  # No shift
        result[:] = arr

    return result

def shift_day_week_to_x(x, num,  weight_cache='', diff_rate = 0.0, return_shifts=False):
    """
    Args:
        x: input data
        num: shift offset
        weight_cache: path to weight cache file
        diff_rate: differential rate
        return_shifts: if True, return the 7 individual shifts instead of weighted average
                      (used for mode='his' to train history weights)
    
    Returns:
        if return_shifts=False: concatenate [x, avg_shift] (normal mode)
        if return_shifts=True: concatenate [x, shifts] where shifts has 7 channels (his mode)
    """
    day_shift = shift_traffic_data(x[...,:1], num =288+num)
    two_shift = shift_traffic_data(x[...,:1], num =576+num)
    threeday_shift = shift_traffic_data(x[...,:1], num =864+num)
    four_shift = shift_traffic_data(x[...,:1], num =1152+num)
    five_shift = shift_traffic_data(x[...,:1], num =1440+num)
    sixday_shift = shift_traffic_data(x[...,:1], num =1728+num)
    week_shift = shift_traffic_data(x[...,:1], num =2016+num)
    
    shifts = np.concatenate([day_shift, two_shift, threeday_shift, four_shift, five_shift, sixday_shift, week_shift], axis= -1)
    
    if return_shifts:
        # For mode='his': return original x concatenated with 7 individual shifts
        shift_x = np.concatenate([x, shifts], axis=-1)
        return shift_x
    else:
        # For normal mode: return original x concatenated with weighted average shift
        avg_shift = np.zeros_like(day_shift)
        shift_weight = np.load(weight_cache).astype(np.float32)
        avg_shift[:2016] = 0.8*day_shift[:2016] + 0.2*week_shift[:2016]
        avg_shift[2016:] = (shifts[2016:] * shift_weight).sum(axis=-1)[...,np.newaxis]
        if diff_rate > 0.0:
            node_x_diff = np.diff(x[...,:1], axis=0)
            node_x_diff_padded = np.zeros_like(x[...,:1])
            node_x_diff_padded[1:] = node_x_diff
            avg_shift = avg_shift + diff_rate * node_x_diff_padded
        shift_x = np.concatenate([x, avg_shift], axis=-1)
        print_log(f"loading his ! shift_x {shift_x.shape} x:{x.shape}, avg_shift:{avg_shift.shape}")
        return shift_x
  

 
def calculate_dtw_grid(data_dir, dataset, geo_ids, num_nodes, len_row, len_column, data_col, use_row_column=True, time_intervals=300, cal_rate=0.8):
    """Calculate DTW matrix for grid data based on non-test set (train + val)"""
    cache_path = data_dir + 'dtw_' + dataset + '.npy'
    
    if os.path.exists(cache_path):
        dtw_matrix = np.load(cache_path)
        print('Loaded DTW matrix from {}'.format(cache_path))
        return dtw_matrix
    
    print('Calculating DTW matrix for grid data (non-test set only)...')
    data, timesolts, idx_of_timesolts = load_grid(data_dir, dataset, geo_ids, len_row, len_column, 
                                                   data_col, load_external=False, use_row_column=use_row_column)
    
    # Only use non-test set portion (train + val)
    cal_len = int(data.shape[0] * cal_rate)
    data_none_test = data[:cal_len]
    
    points_per_hour = 3600 // time_intervals
    
    if use_row_column:
        # 4D data: reshape to (time, nodes, feature)
        time_len, row_len, col_len, feat_dim = data_none_test.shape
        data_reshaped = data_none_test.reshape(time_len, row_len * col_len, feat_dim)
    else:
        data_reshaped = data_none_test
    
    # Calculate mean pattern per day
    data_mean = np.mean(
        [data_reshaped[24 * points_per_hour * i: 24 * points_per_hour * (i + 1)]
         for i in range(data_reshaped.shape[0] // (24 * points_per_hour))], axis=0)
    
    # Use tslearn's cdist_dtw for fast parallel computation
    # data_mean shape: (time, nodes, features) -> transpose to (nodes, time, features)
    data_mean_transposed = data_mean.transpose(1, 0, 2)
    print(f'Computing DTW matrix using tslearn.cdist_dtw for {num_nodes} nodes...')
    dtw_distance = cdist_dtw(data_mean_transposed, n_jobs=-1, verbose=1)
    
    np.save(cache_path, dtw_distance)
    print('Saved DTW matrix to {}'.format(cache_path))
    return dtw_distance

def calculate_dtw_point(data_dir, dataset, geo_ids, num_nodes, data_col, time_intervals=300, cal_rate=0.8):
    """Calculate DTW matrix for point data based on non-test set (train + val)"""
    cache_path = data_dir + 'dtw_' + dataset + '.npy'
    
    if os.path.exists(cache_path):
        dtw_matrix = np.load(cache_path)
        print('Loaded DTW matrix from {}'.format(cache_path))
        return dtw_matrix
    
    print('Calculating DTW matrix for point data (non-test set only)...')
    df = load_dyna(data_path=data_dir, filename=dataset, geo_ids=geo_ids, 
                   load_external=False, data_col=data_col)
    
    # Only use non-test set portion (train + val)
    cal_len = int(df.shape[0] * cal_rate)
    df_none_test = df[:cal_len]
    
    points_per_hour = 3600 // time_intervals
    
    # Calculate mean pattern per day
    # df_none_test shape: (time, nodes, features)
    data_mean = np.mean(
        [df_none_test[24 * points_per_hour * i: 24 * points_per_hour * (i + 1)]
         for i in range(df_none_test.shape[0] // (24 * points_per_hour))], axis=0)
    
    # Use tslearn's cdist_dtw for fast parallel computation
    # data_mean shape: (time, nodes, features) -> transpose to (nodes, time, features)
    data_mean_transposed = data_mean.transpose(1, 0, 2)
    print(f'Computing DTW matrix using tslearn.cdist_dtw for {num_nodes} nodes...')
    dtw_distance = cdist_dtw(data_mean_transposed, n_jobs=-1, verbose=1)
    
    np.save(cache_path, dtw_distance)
    print('Saved DTW matrix to {}'.format(cache_path))
    return dtw_distance

def generate_data_grid(
    data_dir, dataset, data_col, output_dim, batch_size=64, log=None, load_dtw=False, 
    load_external=True, train_rate=0.6, eval_rate=0.2, test_rate=0.2, full_rate=1.0, 
    use_row_column=False, time_intervals=300, log_dir=None,
        in_steps = 6, out_steps = 1, mode='train', weight_adj_epsilon=0.1
):
    """Generate data loaders for grid data"""
    x_list, y_list = [], []
    
    # Load grid geo information
    geo_ids, num_nodes, geo_to_ind, geo_to_rc, len_row, len_column = load_grid_geo(data_dir, dataset)
    
    # Load or calculate DTW matrix
    if load_dtw:
        adj_semx = calculate_dtw_grid(data_dir, dataset, geo_ids, num_nodes, len_row, len_column, 
                                      data_col, use_row_column, time_intervals, 1 - test_rate)
    else:
        dtw_path = data_dir + 'dtw_' + dataset + '.npy'
        if os.path.exists(dtw_path):
            adj_semx = np.load(dtw_path)
        else:
            raise FileNotFoundError(
                f"DTW matrix file not found. Tried:\n"
                f"  - {dtw_path}\n"
                f"  - {dtw_path_old}\n"
                f"Please set load_dtw=True to calculate DTW matrix or provide a pre-computed DTW file."
            )
    
    # Load grid data
    #  use_row_column: (time, row, col, feature) or (time, node, feature)
    df, timesolts, idx_of_timesolts = load_grid(data_dir, dataset, geo_ids, len_row, len_column,
                                                data_col, load_external=load_external, 
                                                use_row_column=use_row_column)
    
    df = df[: round(df.shape[0] * full_rate)].astype(np.float32)
    
    # Load adjacency matrix
    adj_mx, num_nodes = load_grid_rel(data_dir, dataset, num_nodes, len_row, len_column, weight_adj_epsilon=weight_adj_epsilon)
    
    # Add window and horizon
    x, y = add_window_horizon(df, input_window=in_steps, output_window=out_steps)
    
    # Apply shift transformations (same as point data)
    weight_cache = '../finetune_his/' + dataset + '_final_best_hisweight.npy'

    diff_rate = 0.0
    
    print(diff_rate)
    
    # For mode='his': return 7 individual shifts; for other modes: return weighted average
    return_shifts = (mode == 'his')
    x = shift_day_week_to_x(x, -12, weight_cache=weight_cache, diff_rate=diff_rate, return_shifts=return_shifts).astype(np.float32)
    x_list.append(x.astype(np.float32))
    y_list.append(y.astype(np.float32))
    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    y = y[..., :output_dim]
    
    # Split train/val/test
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(
        x, y, train_rate=train_rate, eval_rate=eval_rate, test_rate=test_rate)
    
    # Normalization
    scaler = StandardScaler(mean=x_train[..., :output_dim].mean(), std=x_train[..., :output_dim].std())
    print_log(f"StandardScaler mean: {x_train[..., :output_dim].mean()}, std: {x_train[..., :output_dim].std()}", log=log)
    x_train[..., :output_dim] = scaler.transform(x_train[..., :output_dim])
    x_val[..., :output_dim] = scaler.transform(x_val[..., :output_dim])
    x_test[..., :output_dim] = scaler.transform(x_test[..., :output_dim])
    
    x_train[..., -1:] = scaler.transform(x_train[..., -1:])
    x_val[..., -1:] = scaler.transform(x_val[..., -1:])
    x_test[..., -1:] = scaler.transform(x_test[..., -1:])
    
    print_log(f"Mean:\tx-{scaler.mean}\tStd-{scaler.std}", log=log)
    
    # CDE coefficients
    times = torch.linspace(0, in_steps - 1, in_steps)
    coeff_tra = process_data(x_train[..., :-1], times)
    train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, coeff_tra.transpose(1, 2))
    coeff_val = process_data(x_val[..., :-1], times)
    valid_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, coeff_val.transpose(1, 2))
    coeff_test = process_data(x_test[..., :-1], times)
    test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, coeff_test.transpose(1, 2))
    print_log(f"Traincoeffs:\tx-{train_coeffs[0].shape}\tlen-{len(train_coeffs)}", log=log)
    print_log(f"Valsetcoeffs:  \tx-{valid_coeffs[0].shape}\tlen-{len(valid_coeffs)}", log=log)
    print_log(f"Testsetcoeffs:\tx-{test_coeffs[0].shape}\tlen-{len(test_coeffs)}", log=log)
    
    # Create data loaders
    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train), *train_coeffs
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val), *valid_coeffs
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test), *test_coeffs
    )
    
    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
        num_workers=8, prefetch_factor=4, persistent_workers=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False, pin_memory=True,
        num_workers=4, prefetch_factor=2, persistent_workers=True
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, pin_memory=True,
        num_workers=4, prefetch_factor=2, persistent_workers=True
    )
    
    return trainset_loader, valset_loader, testset_loader, scaler, adj_mx, adj_semx

def generate_data_point(
    data_dir, dataset, data_col, output_dim ,
        tod=False, dow=False, dom=False, batch_size=64, log=None, load_dtw = False,
        load_external=True, train_rate=0.6, eval_rate=0.2, test_rate=0.2, full_rate=1.0,
        input_dim  = 3, log_dir = None, time_intervals=300,
        in_steps = 12, out_steps = 12, mode='train', weight_adj_epsilon=0.1
):  
    x_list, y_list = [], []
    geofile = pd.read_csv(data_dir + str(dataset) + '.geo')
    geo_ids = list(geofile['geo_id'])
    num_nodes = len(geo_ids)
    
    # Load or calculate DTW matrix
    if load_dtw:
        adj_semx = calculate_dtw_point(data_dir, dataset, geo_ids, num_nodes, data_col, time_intervals, 1 - test_rate)
    else:
        dtw_path = data_dir + 'dtw_' + dataset + '.npy'
        if os.path.exists(dtw_path):
            adj_semx = np.load(dtw_path)
            print_log(f'Loaded DTW matrix from {dtw_path}', log=log)
        else:
            raise FileNotFoundError(
                f"DTW matrix file not found. Tried:\n"
                f"  - {dtw_path}\n"
                f"  - {dtw_path_old}\n"
                f"Please set load_dtw=True to calculate DTW matrix or provide a pre-computed DTW file."
            )

    df = load_dyna(data_path=data_dir, filename=str(dataset), geo_ids=geo_ids,  load_external=load_external, data_col=data_col)

    df = df[: round(df.shape[0] * full_rate)].astype(np.float32)
    adj_mx, num_nodes = load_rel(data_path=data_dir, filename=str(dataset), weight_adj_epsilon=weight_adj_epsilon)
    x, y = add_window_horizon(df,input_window= in_steps, output_window= out_steps)
    weight_cache =  '../finetune_his/'+dataset +'_final_best_hisweight.npy'
    
    if dataset == "PEMS04" or dataset == "PEMS03":
        diff_rate = 0.5
    else :
        diff_rate = 0.0
        
    print(diff_rate)
    
    # For mode='his': return 7 individual shifts; for other modes: return weighted average
    return_shifts = (mode == 'his')
    x = shift_day_week_to_x(x, -12,  weight_cache=weight_cache, diff_rate= diff_rate, return_shifts=return_shifts).astype(np.float32)
    x_list.append(x.astype(np.float32))
    y_list.append(y.astype(np.float32))
    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    y = y[..., :output_dim]
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(x, y,train_rate=train_rate, eval_rate= eval_rate, test_rate= test_rate) #划分数据集
    
 
    scaler = StandardScaler(mean=x_train[..., :output_dim].mean(), std=x_train[..., :output_dim].std())
    print_log(f"StandardScaler mean: {x_train[..., :output_dim].mean()}, std: {x_train[..., :output_dim].std()}", log=log) #BTND
    x_train[..., :output_dim] = scaler.transform(x_train[..., :output_dim])
    x_val[..., :output_dim] = scaler.transform(x_val[..., :output_dim])
    x_test[..., :output_dim] = scaler.transform(x_test[..., :output_dim])
    
    x_train[..., -1:] = scaler.transform(x_train[..., -1:])
    x_val[...,-1:] = scaler.transform(x_val[..., -1:])
    x_test[..., -1:] = scaler.transform(x_test[..., -1:])
    #######################shift#####################
    # train_shift = scaler.transform(train_shift)
    # val_shift = scaler.transform(val_shift)
    # test_shift = scaler.transform(test_shift)


    print_log(f"Mean:\tx-{scaler.mean}\tStd-{scaler.std}", log=log)
    #---------------------------------cde---------------------------------
    times = torch.linspace(0, in_steps - 1, in_steps)
    coeff_tra = process_data(x_train[...,:-1], times)
    train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, coeff_tra.transpose(1,2))
    coeff_val = process_data(x_val[...,:-1], times)
    valid_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, coeff_val.transpose(1,2))
    coeff_test = process_data(x_test[...,:-1], times)
    test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, coeff_test.transpose(1,2))
    print_log(f"Traincoeffs:\tx-{train_coeffs[0].shape}\tlen-{len(train_coeffs)}", log=log)
    print_log(f"Valsetcoeffs:  \tx-{valid_coeffs[0].shape}\tlen-{len(valid_coeffs)}", log=log)
    print_log(f"Testsetcoeffs:\tx-{test_coeffs[0].shape}\tlen-{len(test_coeffs)}", log=log)
    #-------------------------loader---------------------------------------------
    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train), *train_coeffs
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val),*valid_coeffs
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test), *test_coeffs
    )

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
        num_workers=8, prefetch_factor=4, persistent_workers=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False, pin_memory=True,
        num_workers=4, prefetch_factor=2, persistent_workers=True
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, pin_memory=True,
        num_workers=4, prefetch_factor=2, persistent_workers=True
    )
    
    return trainset_loader, valset_loader, testset_loader, scaler, adj_mx, adj_semx

def generate_data(
    data_dir, dataset, data_col, output_dim, data_type='point', batch_size=64, log=None, 
    load_dtw=False, load_external=True, train_rate=0.6, eval_rate=0.2, test_rate=0.2, 
    full_rate=1.0, use_row_column=True, time_intervals=300, log_dir=None,  in_steps = 12, out_steps = 12, mode='train',
    weight_adj_epsilon=0.1
):
    """
    Unified entry function for generating data loaders.
    
    Args:
        data_type: 'point' or 'grid', determines which data processing pipeline to use
        use_row_column: Only for grid data, whether to use 4D (True) or 3D (False) format
        time_intervals: Time interval in seconds, used for DTW calculation
        mode: 'train', 'test', 'cal', or 'his'. When mode='his', returns 7 individual shifts for history weight training
        Other parameters are shared between point and grid processing
    
    Returns:
        trainset_loader, valset_loader, testset_loader, scaler, adj_mx, adj_semx
    """
    if data_type == 'grid':
        return generate_data_grid(
            data_dir=data_dir, dataset=dataset, data_col=data_col, output_dim=output_dim,
            batch_size=batch_size, log=log, load_dtw=load_dtw, load_external=load_external,
            train_rate=train_rate, eval_rate=eval_rate, test_rate=test_rate, full_rate=full_rate,
            use_row_column=use_row_column, time_intervals=time_intervals, log_dir=log_dir, in_steps = in_steps, out_steps = out_steps, mode=mode,
            weight_adj_epsilon=weight_adj_epsilon
        )
    elif data_type == 'point':
        return generate_data_point(
            data_dir=data_dir, dataset=dataset, data_col=data_col, output_dim=output_dim,
            batch_size=batch_size, log=log, load_dtw=load_dtw, load_external=load_external,
            train_rate=train_rate, eval_rate=eval_rate, test_rate=test_rate, full_rate=full_rate,
            time_intervals=time_intervals, log_dir=log_dir, in_steps = in_steps, out_steps = out_steps, mode=mode,
            weight_adj_epsilon=weight_adj_epsilon
        )
    else:
        raise ValueError(f"Unknown data_type: {data_type}. Must be 'point' or 'grid'.")



