import numpy as np
import pandas as pd
import os

from scipy.signal import find_peaks, filtfilt

from data_prep.general_functions import load_csv_as_dataframe, get_leg_name_from_file_name
from data_prep.normalize import normalize_value_list


def get_peaks_list(data_list, min_high_of_peaks):
    peaks, _ = find_peaks(data_list, height=min_high_of_peaks)
    np.diff(peaks)
    return np.squeeze(peaks)


def filter_data_list(data_list, filter_b, filter_a, filter_padlen):
    return filtfilt(filter_b, filter_a, data_list, padlen=filter_padlen)


def get_step_data_from_dataframe(data_frame, peak_indexes, data_name):
    return data_frame[data_name].values[peak_indexes[0]:peak_indexes[1]]


def get_step_dataframe_from_haptic_data(data, min_index, max_index, column_names):
    cycle = pd.DataFrame()
    for column_name in column_names:
        step_indexes = (min_index, max_index)
        column_step_values = get_step_data_from_dataframe(data, step_indexes, column_name)
        cycle.insert(len(cycle.columns), column_name, column_step_values)
    cycle.index.name = 'id'
    return cycle


def get_peaks_and_combine_peaks(file_names, root, filter_b, filter_a, filter_padlen_value, min_high_of_peaks):
    combined_peaks_indexes = []
    peaks_and_combine_peaks = {}
    for file_name in file_names:
        haptic_data = load_csv_as_dataframe(os.path.join(root, file_name))
        force_z_values = haptic_data['force_z'].values
        leg_name = get_leg_name_from_file_name(file_name)

        # normalize force_z to <0,1> inside single file to achieve better results in step extraction
        # this normalize doesn't effect data used afterwards, it's only for step finding
        force_z_values = normalize_value_list(force_z_values, min(force_z_values), max(force_z_values))

        # filter force_z to simplify peaks finding
        force_z_values = filter_data_list(force_z_values, filter_b, filter_a, filter_padlen_value)

        # get peaks indexes
        current_peaks_indexes = get_peaks_list(force_z_values, min_high_of_peaks)

        # save in dict by leg name
        peaks_and_combine_peaks[leg_name] = current_peaks_indexes

        # add peaks from current leg to combine list of all legs peaks data
        combined_peaks_indexes[len(combined_peaks_indexes):] = current_peaks_indexes

    # if all legs are done, sort combine list for step length finding purpose
    combined_peaks_indexes.sort()

    # add it to dict that will be return
    peaks_and_combine_peaks['combine'] = combined_peaks_indexes

    # return data of peaks of all legs in given test run and combined list of them
    return peaks_and_combine_peaks
