class MethodResult():
    def __init__(self):
        self.accuracy = None
        self.time_of_execution = None
        self.kernel = None
        self.cache_size = None
        self.C = None
        self.dataset_name = None
        self.clf = None
        self.walk_accuracy = None

    def set(self, dataset_name=None, accuracy=None, time_of_execution=None, kernel=None, cache_size=None,
            c=None, clf=None, walk_accuracy=None):
        self.accuracy = accuracy
        self.time_of_execution = time_of_execution
        self.kernel = kernel
        self.cache_size = cache_size
        self.C = c
        self.dataset_name = dataset_name
        self.clf = clf
        self.walk_accuracy = walk_accuracy

    def return_log_message(self):
        return f'PARAMS FOR: {self.dataset_name} DATASET\nAccuracy: {self.accuracy}\nWalk accuracy: {self.walk_accuracy}\nTime of execution: {self.time_of_execution} \nCache size: {self.cache_size}\nKernel type: {self.kernel}\nC: {self.C}\n'
