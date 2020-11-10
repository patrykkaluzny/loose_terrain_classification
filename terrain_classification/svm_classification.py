from machine_learning.general_functions import check_directory_accessibility, check_and_create_directory, \
    save_pickle, get_walk_dataset_from_inplace_dataset_name
from machine_learning.svm_functions import svm_logs, create_log_file, print_progress, get_svm_cls_acc
from settings import Settings, SVM_param
from machine_learning.svm_method_param_classs import MethodResult
from sklearn.svm import SVC
import os
from sklearn.metrics import accuracy_score
import time
from machine_learning.data_loading import split_dataset, get_features_and_labels_from_path
import numpy as np


def svm_test(datasets_dir_path, test_size=0.2, cache_size=500, kernel='rbf', C=1.0,
             random_state=None):
    check_directory_accessibility(datasets_dir_path, "Dataset directory is not accessible")
    svm_log_file_path = os.path.join(datasets_dir_path, 'log.txt')
    create_log_file(svm_log_file_path)

    for root, dirs, files in os.walk(datasets_dir_path):
        for name in files:
            if 'walk' not in name and '.txt' not in name:
                dataset_path = os.path.join(root, name)

                dataset_name_split = name.split('_')

                walk_dataset_path = os.path.join(root,dataset_name_split[0] + '_balanced_validated_' + dataset_name_split[3] + '_walk')
                features_walk, labels_walk = get_features_and_labels_from_path(walk_dataset_path)

                inpalce_accs = []
                walk_accs = []
                for i in range(0, 10):
                    inpalce_acc, walk_acc = get_svm_cls_acc(dataset_path, features_walk, labels_walk, test_size,
                                                            cache_size, kernel, C, random_state)
                    inpalce_accs.append(inpalce_acc)
                    walk_accs.append(walk_acc)

                log_message = f'Dataset name: {name}\nMeanAccuracy: {np.mean(inpalce_accs)}\nStdAccuracy: {np.std(inpalce_accs)}\nMeanWalkAccuracy: {np.mean(walk_accs)}\nStdWalkAccuracy: {np.std(walk_accs)}\n'
                svm_logs(log_message, svm_log_file_path)


if __name__ == '__main__':
    s = Settings()

    # svm_test('resources/datasets/dwt', test_size=0.2, cache_size=500, kernel='rbf', C=1.0,
    #          random_state=None)
    svm_test('resources/datasets/fft', test_size=0.2, cache_size=500, kernel='rbf', C=1.0,
             random_state=None)
