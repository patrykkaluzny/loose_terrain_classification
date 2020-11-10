from data_prep.general_functions import check_directory_accessibility, check_and_create_directory, abs_dict
from data_prep.normalize import get_min_max_f_t_values, normalize_and_save_data
from settings import Settings


def normalize_raw_data(raw_haptic_data_dir_path, normalize_haptic_data_dir_path):
    # check if raw haptic data is accessible
    check_directory_accessibility(raw_haptic_data_dir_path, 'Raw data directory path unavailable')

    # check if normalize haptic data path is available
    check_and_create_directory(normalize_haptic_data_dir_path,
                               'Problem with creating normalize haptic data directory')

    # get maximum and minimum torque and force value from raw data
    min_max_ft_values = get_min_max_f_t_values(raw_haptic_data_dir_path)
    min_max_ft_values = abs_dict(min_max_ft_values)
    normalize_and_save_data(raw_haptic_data_dir_path, normalize_haptic_data_dir_path,
                            min_max_ft_values)


if __name__ == '__main__':
    s = Settings()
    normalize_raw_data(s.raw_haptic_data_dir_path, s.normalize_haptic_data_dir_path)
