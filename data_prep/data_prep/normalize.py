import os
import pathlib
import pandas as pd
import numpy as np

from data_prep.general_functions import load_csv_as_dataframe, save_dataframe_as_csv


def get_min_max_f_t_values(raw_haptic_data_dir_path):
    min_max_ft_values = {'f_min': 1000, 't_min': 1000, 'f_max': 0, 't_max': 0}
    for root, dirs, files in os.walk(raw_haptic_data_dir_path):
        for name in files:
            if pathlib.Path(name).suffix == '.csv':
                # load csv
                raw_haptic_data = load_csv_as_dataframe(os.path.join(root, name))

                # find min max force and torque in data
                min_max_ft_values = get_series_min_max_values(raw_haptic_data, min_max_ft_values)
    return min_max_ft_values


def get_series_min_max_values(dataframe, min_max_values_dict):
    for series in dataframe.items():
        if series[0].find('force') == 0:
            if series[1].max() > min_max_values_dict['f_max']:
                min_max_values_dict['f_max'] = series[1].max()
            elif series[1].min() < min_max_values_dict['f_min']:
                min_max_values_dict['f_min'] = series[1].min()
        elif series[0].find('torque') == 0:
            if series[1].max() > min_max_values_dict['t_max']:
                min_max_values_dict['t_max'] = series[1].max()
            if series[1].min() < min_max_values_dict['t_min']:
                min_max_values_dict['t_min'] = series[1].min()
    return min_max_values_dict


def normalize_and_save_data(raw_haptic_data_dir_path, norm_haptic_data_dir_path, min_max_ft_values):
    for root, dirs, files in os.walk(raw_haptic_data_dir_path):
        for name in files:
            if pathlib.Path(name).suffix == '.csv':
                # load csv
                haptic_data = load_csv_as_dataframe(os.path.join(root, name))

                # delete old id column
                haptic_data = haptic_data.drop('id', 1)

                # normalize data
                normalize_dataframe(haptic_data, min_max_ft_values)

                # save normalize data
                save_dataframe_as_csv(haptic_data, os.path.join(norm_haptic_data_dir_path, name))


def normalize_dataframe(raw_haptic_data_frame, min_max_ft_values):
    # iterate through all series in dataframe
    for series in raw_haptic_data_frame.iteritems():

        # get series name
        series_name = series[0]

        # check if its force or torque values
        if series_name.find('force') == 0:

            # get series values
            series_values = series[1]

            # normalize data
            norm_force_values_list = normalize_value_list(series_values, min_max_ft_values['f_min'],
                                                          min_max_ft_values['f_max'])

            # create new series from normalize data and series name
            normalize_data = pd.DataFrame({series_name: norm_force_values_list})

            # update data in give dataframe
            raw_haptic_data_frame.update(normalize_data)

        elif series_name.find('torque') == 0:

            # get series values
            series_values = series[1]

            # normalize data [norm = (data+abs(min))/((abs)min+max)]
            # adding abs(min) to normalize from 0 to 1 rather than -1 to 1
            norm_torque_values_list = series_values.add(min_max_ft_values['t_min'])
            norm_torque_values_list = norm_torque_values_list.div(
                min_max_ft_values['t_min'] + min_max_ft_values['t_max'])

            # create new series from normalize data and series name
            normalize_data = pd.DataFrame({series_name: norm_torque_values_list})

            # update data in give dataframe
            raw_haptic_data_frame.update(normalize_data)


def normalize_value_list(value_list, min_value, max_value):
    # normalize data [norm = (data+abs(min))/((abs)min+max)]
    # adding abs(min) to normalize from 0 to 1 rather than -1 to 1
    value_list = np.add(value_list, abs(min_value))

    return np.divide(value_list, abs(min_value) + max_value)


def normalize_dataset(dataset_dict, min_val, max_val):
    normalize_dataset_list = []
    for features_and_label in dataset_dict:
        normalize_dict = {'label': features_and_label['label'],
                          'features': normalize_value_list(features_and_label['features'], min_val, max_val)}
        normalize_dataset_list.append(normalize_dict)
    return normalize_dataset_list

