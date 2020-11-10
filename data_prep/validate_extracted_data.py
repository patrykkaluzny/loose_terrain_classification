from settings import Settings
from data_prep.data_validation import removing_wrong_signals, validate_dataset_dict
from data_prep.general_functions import check_and_create_directory, check_directory_accessibility, save_pickle, \
    load_pickle

import os


def validate_extracted_data(extracted_data_path, validated_dir_path, sample_length_range,
                            samples_percentage, validated_data_name):
    # check if raw haptic data is accessible
    check_directory_accessibility(extracted_data_path, 'extracted data dict unavailable')

    # check if path of save target directory is available, if not create directory
    check_and_create_directory(validated_dir_path, 'Problem with creating dataset directory')

    # load pickle with extracted data
    dataset_dict = load_pickle(extracted_data_path)

    # # print some stats
    # print_stats_of_validated_dataset(dataset_dict, sample_length_range)

    # perform data validation
    stats_dict = validate_dataset_dict(dataset_dict, sample_length_range)
    validated_dataset = removing_wrong_signals(stats_dict, dataset_dict, sample_length_range, samples_percentage)
    # save validated data as pickle in validated data directory
    save_pickle(validated_dir_path, validated_data_name, validated_dataset)


if __name__ == '__main__':
    s = Settings()

    validate_extracted_data(os.path.join(s.extracted_data_dir_path, s.extracted_cycles_file_name),
                            s.validated_data_dir_path, s.cycle_length_range, s.samples_percentage_cycle,
                            s.validated_cycles_file_name)
    validate_extracted_data(os.path.join(s.extracted_data_dir_path, s.extracted_steps_file_name),
                            s.validated_data_dir_path, s.steps_length_range, s.samples_percentage_step,
                            s.validated_steps_file_name)

