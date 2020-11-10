import torch
from torch.utils.data import DataLoader
from machine_learning.general_functions import load_pickle, calculate_valid_dataset_size
from sklearn.model_selection import train_test_split


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, features_list, labels):
        'Initialization'
        self.labels = labels
        self.features_list = features_list

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.features_list)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        features = torch.FloatTensor([self.features_list[index]])
        label = torch.LongTensor([self.labels[index]])

        return features, label


def get_train_test_valid_dataset(path, datasets_sizes, random_state=None):
    train_size = datasets_sizes[0]
    test_size = datasets_sizes[1]
    valid_size = datasets_sizes[2]

    features, labels = get_features_and_labels_from_path(path)
    features_train, features_rest, labels_train, labels_rest = train_test_split(features, labels,
                                                                                test_size=(test_size + valid_size),
                                                                                shuffle=True, random_state=random_state,
                                                                                stratify=labels)
    valid_size = calculate_valid_dataset_size(valid_size, train_size)
    features_test, features_valid, labels_test, labels_valid = train_test_split(features_rest, labels_rest,
                                                                                test_size=valid_size,
                                                                                shuffle=True,
                                                                                random_state=random_state,
                                                                                stratify=labels_rest)

    return features_train, features_test, features_valid, labels_train, labels_test, labels_valid


def get_train_test_dataset(path, test_size, random_state=None):
    features, labels = get_features_and_labels_from_path(path)
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                                test_size=test_size, shuffle=True,
                                                                                random_state=random_state,
                                                                                stratify=labels)

    return features_train, features_test, labels_train, labels_test


def get_features_and_labels_from_path(path):
    dataset = load_pickle(path)
    return dataset['features'].to_list(), dataset['label'].to_list()
    # return dataset['label'].values, dataset['label'].values


def split_dataset(path, datasets_sizes, random_state=None):
    if type(datasets_sizes) is float:
        return get_train_test_dataset(path, datasets_sizes, random_state)
    return get_train_test_valid_dataset(path, datasets_sizes, random_state)


def get_data_loaders_from_dataset_path(dataset_path, dataset_sizes, batch_size, random_state=None):
    x_train, x_test, x_valid, y_train, y_test, y_valid = split_dataset(dataset_path, dataset_sizes, random_state)

    params = {'batch_size': batch_size,
              'shuffle': True,
              'drop_last': True}

    train_set = Dataset(x_train, y_train)
    train_dataloader = torch.utils.data.DataLoader(train_set, **params)

    valid_set = Dataset(x_valid, y_valid)
    valid_dataloader = torch.utils.data.DataLoader(valid_set, **params)

    test_set = Dataset(x_test, y_test)
    test_dataloader = torch.utils.data.DataLoader(test_set, **params)
    return train_dataloader, test_dataloader, valid_dataloader


def get_walk_loader(dataset_path, batch_size):
    x_test, y_test = get_features_and_labels_from_path(dataset_path)
    params = {'batch_size': batch_size,
              'shuffle': True,
              'drop_last': True}

    test_set = Dataset(x_test, y_test)
    test_dataloader = torch.utils.data.DataLoader(test_set, **params)
    return test_dataloader
