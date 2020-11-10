import statistics
from scipy.ndimage import variance


def validate_dataset_dict(dataset, length_range):
    stats_factor_dict = {}
    validated_dict = {}
    for test_name in dataset:
        if test_name != 'walk_21':
            stats_factor_dict[test_name] = {}
            validated_dict[test_name] = {}
            for leg_name in dataset[test_name]:
                stats_factor_dict[test_name][leg_name] = {'variance': []}
                for step in dataset[test_name][leg_name]:
                    for label, content in dataset[test_name][leg_name][step].items():
                        if label == 'force_z':
                            if length_range[0] < content.values.size < length_range[1]:
                                var_value = variance(content.values)
                                stats_factor_dict[test_name][leg_name]['variance'].append(var_value)
    return stats_factor_dict


def min_max_average_calculation(dictionary):
    avg_min = []
    avg_mean = []
    for test_name in dictionary:
        for leg_name in dictionary[test_name]:
            print("Test name: ", test_name, " | Leg name: ", leg_name)
            print("Variance", min(dictionary[test_name][leg_name]['variance']),
                  max(dictionary[test_name][leg_name]['variance']),
                  statistics.mean(dictionary[test_name][leg_name]['variance']))
            avg_min.append(min(dictionary[test_name][leg_name]['variance']))
            avg_mean.append(statistics.mean(dictionary[test_name][leg_name]['variance']))
    return avg_mean, avg_min


def removing_wrong_signals(dictionary, extracted_dataset, length_range, samples_percentage):
    validated_data = {}
    for test_name in extracted_dataset:
        if test_name not in validated_data:
            validated_data[test_name] = {}
        for leg_name in extracted_dataset[test_name]:
            if leg_name not in validated_data[test_name]:
                validated_data[test_name][leg_name] = {}
            if 'walk' in test_name:
                for step_id in extracted_dataset[test_name][leg_name]:
                    force_z = extracted_dataset[test_name][leg_name][step_id]['force_z'].values
                    if length_range[0] < force_z.size < length_range[1]:
                        validated_data[test_name][leg_name][step_id] = extracted_dataset[test_name][leg_name][
                            step_id]
            else:
                var_mean = statistics.mean(dictionary[test_name][leg_name]['variance'])
                for step_id in list(extracted_dataset[test_name][leg_name].keys()):
                    force_z = extracted_dataset[test_name][leg_name][step_id]['force_z'].values
                    var_test = variance(force_z)
                    if var_test > samples_percentage * var_mean or length_range[0] < force_z.size < length_range[1]:
                        validated_data[test_name][leg_name][step_id] = extracted_dataset[test_name][leg_name][
                            step_id]
    return validated_data


def print_stats_of_validated_dataset(dataset_dict, sample_length_range):
    steps_variance = validate_dataset_dict(dataset_dict, sample_length_range)
    avg_mean, avg_min = min_max_average_calculation(steps_variance)
    print("Average of means: ", statistics.mean(avg_mean), " | ", "Average of mins: ", statistics.mean(avg_min))
