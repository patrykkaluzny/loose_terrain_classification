import os
import pathlib
from scipy.signal import butter

from data_prep.general_functions import check_and_create_directory, check_directory_accessibility, \
    get_leg_name_from_file_name, get_test_name_from_file_name, load_csv_as_dataframe, save_pickle
from data_prep.step_extraction import get_step_dataframe_from_haptic_data, get_peaks_list, filter_data_list
from data_prep.normalize import normalize_value_list
from settings import Settings


def extract_cycles(normalize_haptic_data_dir_path, extracted_data_dir_path, extracted_cycles_file_name,
                   min_high_of_peaks, data_column_names, filter_padlen):
    # check if raw haptic data is accessible
    check_directory_accessibility(normalize_haptic_data_dir_path, 'Normalize data directory path unavailable')

    # design filter TODO: check and adjust filters values
    b, a = butter(2, 0.05)

    # create dict to store all extracted steps
    extracted_cycles_dict = {}

    # go through all data files
    for root, dirs, files in os.walk(normalize_haptic_data_dir_path):
        for name in files:
            if pathlib.Path(name).suffix == '.csv':
                # get test name and leg name from file name
                test_name = get_test_name_from_file_name(name)
                leg_name = get_leg_name_from_file_name(name)

                # load dataframe from csv
                haptic_data = load_csv_as_dataframe(os.path.join(root, name))
                force_z_values = haptic_data['force_z'].values

                # create place to store in dict
                if test_name not in extracted_cycles_dict:
                    extracted_cycles_dict[test_name] = {}

                # normalize force_z to <0,1> inside single file to achieve better results in step extraction
                # this normalize doesn't effect data used afterwards, it's only for step finding
                force_z_values = normalize_value_list(force_z_values, min(force_z_values), max(force_z_values))

                # filter force_z to simplify peaks finding
                force_z_values = filter_data_list(force_z_values, b, a, filter_padlen)

                # get peaks indexes
                peaks_indexes = get_peaks_list(force_z_values, min_high_of_peaks)


                # create dict to store single test data
                cycle = {}

                for i in range(1, len(peaks_indexes)):
                    cycle[str(i - 1)] = get_step_dataframe_from_haptic_data(haptic_data, peaks_indexes[i - 1],
                                                                            peaks_indexes[i], data_column_names)

                extracted_cycles_dict[test_name][leg_name] = cycle

    check_and_create_directory(extracted_data_dir_path, 'Problem with creating extract steps data directory')
    save_pickle(extracted_data_dir_path, extracted_cycles_file_name, extracted_cycles_dict)


if __name__ == '__main__':
    s = Settings()
    extract_cycles(s.normalize_haptic_data_dir_path, s.extracted_data_dir_path, s.extracted_cycles_file_name,
                   s.min_high_of_peaks, s.data_column_names, s.filter_padlen)
