import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import os


def load_pickle(pickle_file_path):
    infile = open(pickle_file_path, 'rb')
    return pickle.load(infile)


def calculate_valid_dataset_size(valid_size, train_size):
    # function that calculates how much of rest samples will by assign to valid dataset
    # its calculated as number of samples that left after train set divide normalize to 1
    return valid_size / (1 - train_size)


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


def get_dataset_name_from_path(path):
    return path.split('/')[-1]


def save_pickle(file_path, file_name, obj_to_save):
    outfile = open(os.path.join(file_path, file_name), 'wb')
    pickle.dump(obj_to_save, outfile)


def get_walk_dataset_from_inplace_dataset_name(root_path, inplace_dataset_name):
    return root_path + inplace_dataset_name.split('_')[0] + '_' + inplace_dataset_name.split('_')[1] + '_' + 'walk'
