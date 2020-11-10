def check_minimal_step_number_of_all_tests(dict_of_data):
    min_number = None
    for test_name in dict_of_data:
        if 'walk' not in test_name:
            for i, leg_name in enumerate(dict_of_data[test_name]):
                if min_number is None or min_number > len(dict_of_data[test_name][leg_name]):
                    min_number = len(dict_of_data[test_name][leg_name])
    return min_number
