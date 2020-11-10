from machine_learning.data_loading import split_dataset, get_features_and_labels_from_path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def svm_logs(log_message, log_file_path):
    file_object = open(log_file_path, 'a')
    file_object.write(log_message)
    file_object.close()


def print_progress(number_of_clf_fit, current_clf_fit_number):
    print(f'Progress: {round(number_of_clf_fit/current_clf_fit_number * 100)} %\n')


def create_log_file(log_file_path):
    with open(log_file_path, 'w') as fp:
        fp.close()


def get_svm_cls_acc(dataset_path, features_walk, labels_walk, test_size=0.2, cache_size='500', kernel='rbf', C='1.0',
                    random_state=None):
    # get labeled dataset and split it into train and test set
    features_train, features_test, labels_train, labels_test = split_dataset(dataset_path,
                                                                             test_size)


    # create clf with current params
    clf = SVC(cache_size=cache_size, kernel=kernel, C=C, random_state=random_state)

    clf.fit(features_train, labels_train)

    labels_predict = clf.predict(features_test)
    # get accuracy_inplace of test set
    accuracy_inplace = accuracy_score(labels_test, labels_predict)

    # test on walk dataset
    labels_walk_predict = clf.predict(features_walk)
    # get accuracy_walk of walk set
    accuracy_walk = accuracy_score(labels_walk, labels_walk_predict)
    return accuracy_inplace, accuracy_walk
