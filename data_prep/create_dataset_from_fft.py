from data_prep.general_functions import check_directory_accessibility, check_and_create_directory, load_pickle, \
    get_moisture_from_test_name, save_pickle, find_min_max, get_data_dict_name_from_path
from data_prep.dataset_creation import get_fft_features
from settings import Settings
from data_prep.normalize import normalize_dataset

import os
import pandas as pd


def create_dataset_fft_with_labels(data_dict_path, dataset_data_dir_path, ft_column_names, labels_translate,
                                   fft_length):
    # check if raw haptic data is accessible
    check_directory_accessibility(data_dict_path, 'Validated data dict path unavailable')

    # load dict
    validated_data_dict = load_pickle(data_dict_path)

    # create steps features lists of dict
    features_with_labels = []

    dataset_min, dataset_max = 1000, 0

    # iterate through all steps
    for test_name in validated_data_dict:
        for leg_name in validated_data_dict[test_name]:
            for step_id in validated_data_dict[test_name][leg_name]:

                # assign step dataframe to be more readable in further code
                step_dataframe = validated_data_dict[test_name][leg_name][step_id]

                # create step features dict
                step_features_and_label = {'label': None, 'features': []}

                # extract fft data from 6 f/t signals and add it to feature vector of step
                feature_vector_list = []
                for data_label in ft_column_names:
                    if data_label != 'time':
                        fft_features = get_fft_features(step_dataframe[data_label].values, fft_length)
                        if fft_features is not None:
                            dataset_min, dataset_max = find_min_max(dataset_min, dataset_max, fft_features)
                            feature_vector_list.extend(fft_features)

                # if its not walk file add labels to features vector
                # add features vectors to right dict
                if len(feature_vector_list) != 0:
                    step_features_and_label['features'] = feature_vector_list
                    step_features_and_label['label'] = labels_translate[get_moisture_from_test_name(test_name)]
                    features_with_labels.append(step_features_and_label)

    # check if dataset directory path is available
    check_and_create_directory(dataset_data_dir_path, 'Problem with creating dataset directory')

    # normalize features
    norm_inplace_features_with_labels = pd.DataFrame(
        normalize_dataset(features_with_labels, dataset_min, dataset_max))

    # save fft features datasets
    data_name = get_data_dict_name_from_path(data_dict_path)
    save_pickle(dataset_data_dir_path, 'fft_' + data_name, norm_inplace_features_with_labels)


if __name__ == '__main__':
    s = Settings()
    for root, dirs, files in os.walk(s.balanced_data_dir_path):
        for name in files:
            create_dataset_fft_with_labels(os.path.join(root, name), s.datasets_dir_path, s.ft_column_names,
                                           s.labels_translate_from_moisture, s.fft_length)
