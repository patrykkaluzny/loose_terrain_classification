from data_prep.general_functions import check_directory_accessibility, check_and_create_directory, load_pickle, \
    save_pickle, get_data_dict_name_from_path
from data_prep.split_and_balance import check_minimal_step_number_of_all_tests
from settings import Settings
from data_prep.normalize import normalize_dataset

import os
import pandas as pd


def split_and_balance_data(inplace_walk_data_path, balanced_data_dir_path):
    # check if raw haptic data is accessible
    check_directory_accessibility(inplace_walk_data_path, 'Provided data dict path unavailable')

    # load dict
    data_dict = load_pickle(inplace_walk_data_path)

    walk_data_dict = {}
    inplace_data_dict = {}

    # check minimal number of steps in tests to provide balanced datasets
    min_number_of_samples = check_minimal_step_number_of_all_tests(data_dict)

    # iterate through all steps
    for test_name in data_dict:
        if test_name not in walk_data_dict:
            if 'walk' in test_name:
                walk_data_dict[test_name] = {}
                for leg_name in data_dict[test_name]:
                    walk_data_dict[test_name][leg_name] = data_dict[test_name][leg_name]
            else:
                inplace_data_dict[test_name] = {}
                for leg_name in data_dict[test_name]:
                    if leg_name not in inplace_data_dict[test_name]:
                        inplace_data_dict[test_name][leg_name] = {}
                        for i, step_id in enumerate(data_dict[test_name][leg_name]):
                            if i < min_number_of_samples:
                                inplace_data_dict[test_name][leg_name][step_id] = data_dict[test_name][leg_name][
                                    step_id]

    # check if dataset directory path is available
    check_and_create_directory(balanced_data_dir_path, 'Problem with creating dataset directory')

    # get data file name from file
    data_name = get_data_dict_name_from_path(inplace_walk_data_path)

    # save dict as pickles
    save_pickle(balanced_data_dir_path, 'balanced_' + data_name + '_walk', walk_data_dict)
    save_pickle(balanced_data_dir_path, 'balanced_' + data_name + '_inplace', inplace_data_dict)


if __name__ == '__main__':
    s = Settings()
    # iterate through all extracted data
    for root, dirs, files in os.walk(s.extracted_data_dir_path):
        for name in files:
            split_and_balance_data(os.path.join(root, name), s.balanced_data_dir_path)

    # iterate through all validated data
    for root, dirs, files in os.walk(s.validated_data_dir_path):
        for name in files:
            split_and_balance_data(os.path.join(root, name), s.balanced_data_dir_path)
