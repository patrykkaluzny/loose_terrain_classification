import os
import pathlib

from scipy.signal import butter

from data_prep.general_functions import check_and_create_directory, check_directory_accessibility, \
    get_leg_name_from_file_name, get_test_name_from_file_name, get_all_test_names, get_all_files_names_from_single_test, \
    load_csv_as_dataframe, save_pickle
from data_prep.step_extraction import get_peaks_and_combine_peaks, get_step_dataframe_from_haptic_data
from settings import Settings


def extract_steps(normalize_haptic_data_dir_path, extracted_data_dir_path, extracted_steps_file_name, min_high_of_peaks,
                  data_column_names, filter_padlen):
    # check if raw haptic data is accessible
    check_directory_accessibility(normalize_haptic_data_dir_path, 'Normalize data directory path unavailable')

    # create dict to store peaks indexes extracted from filtered normalized data
    peaks_indexes = {}

    # design filter TODO: check and adjust filters values
    b, a = butter(2, 0.05)

    # create dict to store all extracted steps
    extracted_steps_dict = {}

    # get all test names to collect peaks data from all legs of one test
    test_names = get_all_test_names(normalize_haptic_data_dir_path)

    # go through all data files but with test names in mind
    for test_name in test_names:
        for root, dirs, files in os.walk(normalize_haptic_data_dir_path):
            # save peaks indexes data of all legs from one test and combined sorted list of them
            # if you have combine peaks data, you cen extract step data from beginning of step to start next step of
            # any leg, rather than step data to start new step of the same leg
            single_test_files_list = get_all_files_names_from_single_test(files, test_name)
            peaks_indexes[test_name] = get_peaks_and_combine_peaks(single_test_files_list, root, b, a, filter_padlen,
                                                                   min_high_of_peaks)

    # go through all data files to extract steps with data of all peaks in single test run from all legs
    for root, dirs, files in os.walk(normalize_haptic_data_dir_path):
        for name in files:
            if pathlib.Path(name).suffix == '.csv':

                # get test name and leg name from file name
                test_name = get_test_name_from_file_name(name)
                leg_name = get_leg_name_from_file_name(name)

                # load data from file
                haptic_data = load_csv_as_dataframe(os.path.join(root, name))

                # create place to store in dict
                if test_name not in extracted_steps_dict:
                    extracted_steps_dict[test_name] = {}

                # create dict to store single test data
                steps = {}

                # get peaks list of chosen leg from dict created before
                current_peaks_indexes = peaks_indexes[test_name][leg_name]
                # get peaks list of all legs from chosen test run from dict created before
                current_combined_cycle_peaks_indexes = peaks_indexes[test_name]['combine']

                # go through current leg peaks
                for i, peak_index in enumerate(current_peaks_indexes):

                    # find current peak in combined peaks data
                    index_position_in_combine_list = current_combined_cycle_peaks_indexes.index(peak_index)

                    # check if not out of space
                    if index_position_in_combine_list + 1 < len(current_combined_cycle_peaks_indexes):
                        # if not end point of step will be next index of current peak in combine list
                        end_peak_index = current_combined_cycle_peaks_indexes[index_position_in_combine_list + 1]

                        # save step data to dict of steps
                        steps[str(i - 1)] = get_step_dataframe_from_haptic_data(haptic_data, peak_index,
                                                                                end_peak_index, data_column_names)

                # save steps of current test and leg to dict
                extracted_steps_dict[test_name][leg_name] = steps

                # check saving path
                check_and_create_directory(extracted_data_dir_path,
                                           'Problem with creating extract steps data directory')

                # save extracted steps data as pickle
                save_pickle(extracted_data_dir_path, extracted_steps_file_name, extracted_steps_dict)


if __name__ == '__main__':
    s = Settings()
    extract_steps(s.normalize_haptic_data_dir_path, s.extracted_data_dir_path, s.extracted_steps_file_name,
                  s.min_high_of_peaks, s.data_column_names, s.filter_padlen)
