import os
import pandas as pd
import pickle
import pathlib


def check_directory_accessibility(directory_path, error_message=''):
    if not os.path.exists(directory_path):
        print(error_message)
        exit()


def check_and_create_directory(directory_path, error_message=''):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
        except NameError:
            print(error_message)
            exit()


def save_dataframe_as_csv(dataframe, file_path_to_save):
    dataframe.index.name = 'id'
    dataframe.to_csv(file_path_to_save)


def save_dict_as_csv(dictionary, file_path_to_save):
    dataframe = pd.DataFrame(dictionary)
    save_dataframe_as_csv(dataframe, file_path_to_save)


def load_csv_as_dataframe(path):
    return pd.read_csv(path)


def abs_dict(dictionary):
    return {x: abs(dictionary[x]) for x in dictionary}


def get_test_name_from_file_name(file_name):
    name_without_extension = file_name.split('.')[0]
    return name_without_extension.split('_')[0] + '_' + name_without_extension.split('_')[1]


def get_leg_name_from_file_name(file_name):
    name_without_extension = file_name.split('.')[0]
    return name_without_extension.split('_')[3]


def save_pickle(file_path, file_name, obj_to_save):
    outfile = open(os.path.join(file_path, file_name), 'wb')
    pickle.dump(obj_to_save, outfile)


def load_pickle(pickle_file_path):
    infile = open(pickle_file_path, 'rb')
    return pickle.load(infile)


def get_all_test_names(dir_path):
    tests_names = []
    for root, dirs, files in os.walk(dir_path):
        for name in files:
            if pathlib.Path(name).suffix == '.csv':
                test_name = get_test_name_from_file_name(name)
                if test_name not in tests_names:
                    tests_names.append(test_name)
    return tests_names


def get_all_files_names_from_single_test(all_files_names, desired_file_name):
    return [x for x in all_files_names if x.find(desired_file_name) == 0]


def get_moisture_from_test_name(test_name):
    return test_name.split('_')[1]


def get_data_type_name_from_path(path):
    return path.split('_')[-1]


def find_min_max(min_val, max_val, signal):
    if min_val > min(signal):
        min_val = min(signal)
    if max_val < max(signal):
        max_val = max(signal)
    return min_val, max_val


def get_data_dict_name_from_path(path):
    return path.split('/')[-1]
