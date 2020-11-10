import numpy as np


class Settings:
    """A class to store all settings of terrain_classification project"""

    def __init__(self):
        # paths
        self.datasets_dir_path = 'resources/datasets/'
        self.models_dir_path = 'resources/models/'

        # datasets splitting variables
        self.random_state = 42
        self.test_size = 0.2
        self.datasets_sizes = (0.6, 0.2, 0.2)

        self.svm_log_file_path = 'resources/svm_log.txt'
        self.svm_best_log_file_path = 'resources/best_svm_log.txt'
        self.svm_best_walk_log_file_path = 'resources/best_walk_svm_log.txt'

        self.batch_size = 16
        self.step_length = 150
        self.cycle_length = 500


class SVM_param:

    def __init__(self):
        self.cache_size = [200, 500, 1000]  # more is better, slows compute
        self.kernel = ['rbf', 'poly', 'linear', 'sigmoid']  # different kernel types
        self.C = [0.1, 0.5, 1.0, 2, 5, 10]  # regularization parameter

        # simplify data for tests
        # self.cache_size = [200]  # more is better, slows compute
        # self.kernel = ['rbf']  # different kernel types
        # self.C = [1.0]  # regularization parameter
