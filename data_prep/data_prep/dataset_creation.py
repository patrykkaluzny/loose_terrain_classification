import numpy as np
from scipy.stats import kurtosis, skew
from scipy.ndimage import variance
import pywt


def calculate_fft_return_abs(data):
    y = np.fft.fft(data)
    return np.abs(y[:len(y) // 2])


def get_fft_features(data, desire_length):
    if desire_length * 2 > len(data):
        print('wrong fft length')
        return None
    else:
        y = calculate_fft_return_abs(data)

    # return values without fits one to prevent data other than first to be relevant
    return y[1:desire_length + 1]


def get_kurtosis(data):
    return kurtosis(data)


def get_skew(data):
    return skew(data)


def get_variance(data):
    return variance(data)


def get_sp_vector(data):
    return [get_variance(data), get_skew(data), get_kurtosis(data)]


def get_padded_feature_vector_from_ft(step_data, padding_size):
    feature_vector = []
    for columnName, columnData in step_data.iteritems():
        if columnName != 'time':
            feature_vector.extend(perform_padding(columnData.values, padding_size))
    return feature_vector


def perform_padding(list_of_values, padding_size):
    if len(list_of_values) > padding_size:
        return list_of_values[:padding_size]
    padded_data = np.zeros(padding_size)
    padded_data[:len(list_of_values)] = list_of_values
    return padded_data.tolist()


def get_dwt_features(data, wavelet, mode, level):
    coeff_list = pywt.wavedec(data, wavelet, mode=mode, level=level)
    features_list = []
    print(f'level 1 len: {len(coeff_list[0])}, level 2 len: {len(coeff_list[1])}')
    for i in range(1, level + 1):
        features_list.extend(coeff_list[i])
    return features_list
