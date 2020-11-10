from machine_learning.general_functions import check_directory_accessibility
from machine_learning.lstm_functions import LSTMClassifier, CyclicLR, cosine
from machine_learning.data_loading import get_data_loaders_from_dataset_path, get_walk_loader
from settings import Settings
import torch
from torch import nn
from torch.nn import functional as F
import os
import numpy as np


def lstm_logs(log_message, log_file_path):
    file_object = open(log_file_path, 'a')
    file_object.write(log_message)
    file_object.close()


def testing_lstm(test_dl, model_path, dataset, input_dim=3000, hidden_dim=256, layer_dim=2, output_dim=6):
    model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim)
    model.load_state_dict(torch.load(model_path + dataset + '_lstm_model'))
    model.cuda()
    model.eval()
    correct, total = 0, 0

    for _, single_data in enumerate(test_dl):
        x_test, y_test = single_data[0].cuda(), single_data[1].cuda()
        out = model(x_test)
        # print("Output from model: ", out)
        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        y_test = y_test.squeeze()
        # print('Model prediction: ', preds, "y_val:  ", y_test)
        total += y_test.size(0)
        correct += (preds == y_test).sum().item()
    acc = correct / total
    print(f'Test Acc.: {acc:2.2%}')
    return acc


def training_lstm(dataset, train_dl, valid_dl, models_dir_path, input_dim=3000, hidden_dim=256, layer_dim=2, output_dim=6):
    n_epochs = 1000
    lr = 0.0005
    iterations_per_epoch = len(train_dl)
    best_acc = 0
    best_loss = 100
    patience, trials = 10, 0

    model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.RMSprop(model.parameters(), lr=lr)
    sched = CyclicLR(opt, cosine(t_max=iterations_per_epoch * 2, eta_min=lr / 100))

    print('Start model training')

    for epoch in range(1, n_epochs + 1):

        for i, single_data in enumerate(train_dl):
            model.train()
            x_batch, y_batch = single_data[0].cuda(), single_data[1].cuda()
            y_batch = y_batch.squeeze()
            opt.step()
            sched.step()
            opt.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()

        model.eval()
        correct, total = 0, 0
        for _, single_data in enumerate(valid_dl):
            x_val, y_val = single_data[0].cuda(), single_data[1].cuda()
            out = model(x_val)
            preds = F.log_softmax(out).argmax(1)
            y_val = y_val.squeeze()
            total += y_val.size(0)
            correct += (preds == y_val).sum().item()

        acc = correct / total
        # print(f'Epoch: {epoch:3d}. Loss: {loss.item()}. Acc.: {acc:2.2%}')

        if acc > best_acc or loss.item() < best_loss:
            trials = 0
            best_acc = acc
            best_loss = loss.item()
        else:
            trials += 1
            if trials >= patience:
                print(f'Early stopping on epoch {epoch}')
                torch.save(model.state_dict(), models_dir_path + dataset + '_lstm_model')
                print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
                break


def lstm_classification(settings):
    datasets_names = ['raw_balanced_extracted_cycles_inplace',
                      'raw_balanced_extracted_steps_inplace',
                      'raw_balanced_validated_cycles_inplace',
                      'raw_balanced_validated_steps_inplace']
    walk_names = ['raw_balanced_validated_cycles_walk',
                  'raw_balanced_validated_steps_walk']
    check_directory_accessibility(settings.datasets_dir_path, "Dataset directory is not accessible")
    lstm_log_file_path = os.path.join(settings.datasets_dir_path, 'lstm_log.txt')
    for dataset in datasets_names:
        if dataset == 'raw_balanced_extracted_cycles_inplace':
            dataset_path = os.path.join(settings.datasets_dir_path, dataset)
            walk_path = os.path.join(settings.datasets_dir_path, walk_names[0])
            train_dl, test_dl, valid_dl = get_data_loaders_from_dataset_path(dataset_path, s.datasets_sizes, s.batch_size, random_state=None)
            walk_dl = get_walk_loader(walk_path, s.batch_size)
            params = {
            'input_dim': 3000,
            'hidden_dim': 256,
            'layer_dim': 4,
            'output_dim': 6
            }
            acc_list = []
            walk_list = []
            for i in range(0, 10):
                training_lstm(dataset, train_dl, valid_dl, settings.models_dir_path, **params)

                acc = testing_lstm(test_dl, settings.models_dir_path, dataset, **params)
                acc_list.append(acc)
                walk = testing_lstm(walk_dl, settings.models_dir_path, dataset, **params)
                walk_list.append(walk)
            log_message = f'Dataset name: {dataset}\nMeanAccuracy: {np.mean(acc_list)*100}\nStdAccuracy: {np.std(acc_list)*100}' \
                          f'\nMeanWalkAccuracy: {np.mean(walk_list)*100}\nStdWalkAccuracy: {np.std(walk_list)*100}\n'
            lstm_logs(log_message, lstm_log_file_path)
            print("Finished")

        elif dataset == 'raw_balanced_extracted_steps_inplace':
            dataset_path = os.path.join(settings.datasets_dir_path, dataset)
            walk_path = os.path.join(settings.datasets_dir_path, walk_names[1])
            train_dl, test_dl, valid_dl = get_data_loaders_from_dataset_path(dataset_path, s.datasets_sizes,
                                                                             s.batch_size, random_state=None)
            walk_dl = get_walk_loader(walk_path, s.batch_size)
            params = {
                'input_dim': 900,
                'hidden_dim': 256,
                'layer_dim': 4,
                'output_dim': 6
            }
            acc_list = []
            walk_list = []
            for i in range(0, 10):
                training_lstm(dataset, train_dl, valid_dl, settings.models_dir_path, **params)

                acc = testing_lstm(test_dl, settings.models_dir_path, dataset, **params)
                acc_list.append(acc)
                walk = testing_lstm(walk_dl, settings.models_dir_path, dataset, **params)
                walk_list.append(walk)
            log_message = f'Dataset name: {dataset}\nMeanAccuracy: {np.mean(acc_list)*100}\nStdAccuracy: {np.std(acc_list)*100}' \
                          f'\nMeanWalkAccuracy: {np.mean(walk_list)*100}\nStdWalkAccuracy: {np.std(walk_list)*100}\n'
            lstm_logs(log_message, lstm_log_file_path)
            print("Finished")

        elif dataset == 'raw_balanced_validated_cycles_inplace':
            dataset_path = os.path.join(settings.datasets_dir_path, dataset)
            walk_path = os.path.join(settings.datasets_dir_path, walk_names[0])
            train_dl, test_dl, valid_dl = get_data_loaders_from_dataset_path(dataset_path, s.datasets_sizes,
                                                                             s.batch_size, random_state=None)
            walk_dl = get_walk_loader(walk_path, s.batch_size)
            params = {
                'input_dim': 3000,
                'hidden_dim': 256,
                'layer_dim': 4,
                'output_dim': 6
            }
            acc_list = []
            walk_list = []
            for i in range(0, 10):
                training_lstm(dataset, train_dl, valid_dl, settings.models_dir_path, **params)

                acc = testing_lstm(test_dl, settings.models_dir_path, dataset, **params)
                acc_list.append(acc)
                walk = testing_lstm(walk_dl, settings.models_dir_path, dataset, **params)
                walk_list.append(walk)
            log_message = f'Dataset name: {dataset}\nMeanAccuracy: {np.mean(acc_list)*100}\nStdAccuracy: {np.std(acc_list)*100}' \
                          f'\nMeanWalkAccuracy: {np.mean(walk_list)*100}\nStdWalkAccuracy: {np.std(walk_list)*100}\n'
            lstm_logs(log_message, lstm_log_file_path)
            print("Finished")

        elif dataset == 'raw_balanced_validated_steps_inplace':
            dataset_path = os.path.join(settings.datasets_dir_path, dataset)
            walk_path = os.path.join(settings.datasets_dir_path, walk_names[1])
            train_dl, test_dl, valid_dl = get_data_loaders_from_dataset_path(dataset_path, s.datasets_sizes,
                                                                             s.batch_size, random_state=None)
            walk_dl = get_walk_loader(walk_path, s.batch_size)
            params = {
                'input_dim': 900,
                'hidden_dim': 256,
                'layer_dim': 4,
                'output_dim': 6
            }
            acc_list = []
            walk_list = []
            for i in range(0, 10):
                training_lstm(dataset, train_dl, valid_dl, settings.models_dir_path, **params)

                acc = testing_lstm(test_dl, settings.models_dir_path, dataset, **params)
                acc_list.append(acc)
                walk = testing_lstm(walk_dl, settings.models_dir_path, dataset, **params)
                walk_list.append(walk)
            log_message = f'Dataset name: {dataset}\nMeanAccuracy: {np.mean(acc_list)*100}\nStdAccuracy: {np.std(acc_list)*100}' \
                          f'\nMeanWalkAccuracy: {np.mean(walk_list)*100}\nStdWalkAccuracy: {np.std(walk_list)*100}\n'
            lstm_logs(log_message, lstm_log_file_path)
            print("Finished")


if __name__ == '__main__':
    s = Settings()

    # perform lstm classification
    lstm_classification(s)
