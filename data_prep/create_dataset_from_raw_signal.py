from data_prep.general_functions import check_directory_accessibility, check_and_create_directory, load_pickle, \
    get_moisture_from_test_name, save_pickle, get_data_dict_name_from_path
from data_prep.dataset_creation import get_padded_feature_vector_from_ft, perform_padding
from settings import Settings

import os
import pandas as pd


def create_dataset_raw_signal_with_labels(data_dict_path, dataset_data_dir_path, labels_translate, padding_size):
    # check if raw haptic data is accessible
    check_directory_accessibility(data_dict_path, 'Validated data dict path unavailable')

    # load dict
    data_dict = load_pickle(data_dict_path)

    # create steps features lists of dict
    features_with_labels = []

    # iterate through all steps
    for test_name in data_dict:
        for leg_name in data_dict[test_name]:
            for step_id in data_dict[test_name][leg_name]:
                step_dict = data_dict[test_name][leg_name][step_id]
                # create step features dict
                padded_data = get_padded_feature_vector_from_ft(step_dict, padding_size)
                label = [labels_translate[get_moisture_from_test_name(test_name)]]
                step_features_and_label = {'label': label,
                                           'features': padded_data}

                # if its not walk file add labels to features vector
                # add features vectors to right dict
                features_with_labels.append(step_features_and_label)

    # check if dataset directory path is available
    check_and_create_directory(dataset_data_dir_path, 'Problem with creating dataset directory')

    # save fft features datasets
    features_with_labels = pd.DataFrame(features_with_labels)
    data_name = get_data_dict_name_from_path(data_dict_path)
    save_pickle(dataset_data_dir_path, 'raw_' + data_name, features_with_labels)


if __name__ == '__main__':
    s = Settings()

    for root, dirs, files in os.walk(s.balanced_data_dir_path):
        for name in files:
            if 'step' in name:
                create_dataset_raw_signal_with_labels(os.path.join(root, name), s.datasets_dir_path,
                                                      s.labels_translate_from_moisture, s.step_padding_size)
            else:
                create_dataset_raw_signal_with_labels(os.path.join(root, name), s.datasets_dir_path,
                                                      s.labels_translate_from_moisture, s.cycle_padding_size)
