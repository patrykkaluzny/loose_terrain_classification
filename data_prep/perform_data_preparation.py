import argparse
import os
from settings import Settings
from data_prep.general_functions import check_and_create_directory
from read_from_rosbag import read_from_rosbag
from normalize_raw_data import normalize_raw_data
from extract_cycles import extract_cycles
from extract_steps import extract_steps
from validate_extracted_data import validate_extracted_data
from split_and_balance_data import split_and_balance_data
from create_dataset_from_dwt import create_dataset_dwt_with_labels
from create_dataset_from_fft import create_dataset_fft_with_labels
from create_dataset_from_signal_paramaters import create_dataset_sp_with_labels
from create_dataset_from_raw_signal import create_dataset_raw_signal_with_labels


parser = argparse.ArgumentParser(description='Data preparation procedure')

parser.add_argument('--rosbags_dir', type=str, default='/media/patryk/2EF0EB06F0EAD35F/Studia/magisterka/a8_tests',
                    help='rosbag files directory path, no default value')

parser.add_argument('--results_dir', type=str, default=os.getcwd(),
                    help='results files directory path, default value is current path')

args = parser.parse_args()

resources_dir_path = os.path.join(args.results_dir, 'resources')
check_and_create_directory(resources_dir_path)

raw_data_dir_path = os.path.join(resources_dir_path, 'raw_data')
norm_data_dir_path = os.path.join(resources_dir_path, 'normalize_data')
balanced_data_dir_path = os.path.join(resources_dir_path, 'balanced_data')
extracted_data_dir_path = os.path.join(resources_dir_path, 'extracted_data')
valid_data_dir_path = os.path.join(resources_dir_path, 'validated_data')
datasets_dir_path = os.path.join(resources_dir_path, 'datasets')

s = Settings(args.rosbags_dir, raw_data_dir_path, norm_data_dir_path, datasets_dir_path, extracted_data_dir_path,
             valid_data_dir_path, balanced_data_dir_path)

print('Starting data preparation procedure')

read_from_rosbag(s.rosbag_dir_path, s.raw_haptic_data_dir_path, s.legs_names, s.leg_names_translate,
                     s.force_torque_offsets)

print('Reading data from rosbag done')

normalize_raw_data(s.raw_haptic_data_dir_path, s.normalize_haptic_data_dir_path)

print('Data normalization done')

extract_cycles(s.normalize_haptic_data_dir_path, s.extracted_data_dir_path, s.extracted_cycles_file_name,
                   s.min_high_of_peaks, s.data_column_names, s.filter_padlen)

print('Cycles extraction done')

extract_steps(s.normalize_haptic_data_dir_path, s.extracted_data_dir_path, s.extracted_steps_file_name,
                  s.min_high_of_peaks, s.data_column_names, s.filter_padlen)

print('Steps extraction done')

validate_extracted_data(os.path.join(s.extracted_data_dir_path, s.extracted_cycles_file_name),
                        s.validated_data_dir_path, s.cycle_length_range, s.samples_percentage_cycle,
                        s.validated_cycles_file_name)

print('Cycles validation done')

validate_extracted_data(os.path.join(s.extracted_data_dir_path, s.extracted_steps_file_name),
                        s.validated_data_dir_path, s.steps_length_range, s.samples_percentage_step,
                        s.validated_steps_file_name)

print('Steps validation done')

for root, dirs, files in os.walk(s.extracted_data_dir_path):
    for name in files:
        split_and_balance_data(os.path.join(root, name), s.balanced_data_dir_path)

print('Cycles balancing done')

# iterate through all validated data
for root, dirs, files in os.walk(s.validated_data_dir_path):
    for name in files:
        split_and_balance_data(os.path.join(root, name), s.balanced_data_dir_path)

print('Steps balancing done')

for root, dirs, files in os.walk(s.balanced_data_dir_path):
    for name in files:
        create_dataset_sp_with_labels(os.path.join(root, name), s.datasets_dir_path, s.ft_column_names,
                                      s.labels_translate_from_moisture)

print('SP datasets created')

for root, dirs, files in os.walk(s.balanced_data_dir_path):
    for name in files:
        create_dataset_fft_with_labels(os.path.join(root, name), s.datasets_dir_path, s.ft_column_names,
                                       s.labels_translate_from_moisture, s.fft_length)

print('FFT datasets created')

for root, dirs, files in os.walk(s.balanced_data_dir_path):
    for name in files:
        if 'step' in name:
            create_dataset_dwt_with_labels(os.path.join(root, name), s.datasets_dir_path, s.ft_column_names,
                                           s.labels_translate_from_moisture, s.dwt_wavelet, s.dwt_mode, s.dwt_level,
                                           s.step_padding_size)
        else:
            create_dataset_dwt_with_labels(os.path.join(root, name), s.datasets_dir_path, s.ft_column_names,
                                           s.labels_translate_from_moisture, s.dwt_wavelet, s.dwt_mode, s.dwt_level,
                                           s.cycle_padding_size)

print('DWT datasets created')

for root, dirs, files in os.walk(s.balanced_data_dir_path):
    for name in files:
        if 'step' in name:
            create_dataset_raw_signal_with_labels(os.path.join(root, name), s.datasets_dir_path,
                                                  s.labels_translate_from_moisture, s.step_padding_size)
        else:
            create_dataset_raw_signal_with_labels(os.path.join(root, name), s.datasets_dir_path,
                                                  s.labels_translate_from_moisture, s.cycle_padding_size)

print('Raw signals datasets created')




