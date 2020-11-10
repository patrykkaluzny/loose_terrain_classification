class Settings():
    """A class to store all settings of data_preparation project"""

    def __init__(self, rosbag_dir_path = '/media/patryk/2EF0EB06F0EAD35F/Studia/magisterka/a8_tests',
                 raw_haptic_data_dir_path='resources/raw_haptic_data',
                 normalize_haptic_data_dir_path='resources/normalize_haptic_data',
                 datasets_dir_path='resources/datasets/',
                 extracted_data_dir_path='resources/extracted_data',
                 validated_data_dir_path='resources/validated_data',
                 balanced_data_dir_path='resources/balanced_data'
                 ):
        # paths
        self.rosbag_dir_path = rosbag_dir_path
        self.raw_haptic_data_dir_path = raw_haptic_data_dir_path
        self.normalize_haptic_data_dir_path = normalize_haptic_data_dir_path
        self.datasets_dir_path = datasets_dir_path
        self.extracted_data_dir_path = extracted_data_dir_path
        self.validated_data_dir_path = validated_data_dir_path
        self.balanced_data_dir_path = balanced_data_dir_path

        # variables used in dataset creation
        self.data_column_names = ('time', 'force_x', 'force_y', 'force_z', 'torque_x', 'torque_y', 'torque_z')
        self.ft_column_names = ['force_x', 'force_y', 'force_z', 'torque_x', 'torque_y', 'torque_z']
        self.legs_names = ('lf', 'lh', 'rf', 'rh')
        self.leg_names_translate = {'ROKUB_1': 'rh', 'ROKUB_2': 'lf', 'ROKUB_3': 'rf', 'ROKUB_4': 'lh'}
        self.force_torque_offsets = {
            'lf': {'force_x': -8.65969657898, 'force_y': -13.5792722702, 'force_z': -9.15240097046,
                   'torque_x': -0.28442505002, 'torque_y': 0.332609713078,
                   'torque_z': 0.249344989657},
            'rf': {'force_x': -4.97796440125, 'force_y': 16.2728500366, 'force_z': 6.4284901619,
                   'torque_x': -0.369241416454, 'torque_y': -0.0522938333452,
                   'torque_z': -0.017724942416},
            'lh': {'force_x': -11.7581958771, 'force_y': -3.56582927704, 'force_z': -12.4880580902,
                   'torque_x': -0.134532049298, 'torque_y': 0.366167902946,
                   'torque_z': 0.183928474784},
            'rh': {'force_x': -10.9087543488, 'force_y': -4.97502803802, 'force_z': -6.74133253098,
                   'torque_x': 0.0321601033211, 'torque_y': 0.224655479193,
                   'torque_z': -0.0646027475595}}

        self.min_high_of_peaks = 0.9
        self.min_distance_between_peaks = 100
        self.calculated_cycle_length = 330
        self.calculated_step_length = 85
        self.filter_padlen = 50

        self.samples_percentage_cycle = 0.75
        self.samples_percentage_step = 0.6
        self.cycle_length_range = (100, 500)
        self.steps_length_range = (60, 180)

        self.labels_translate_from_moisture = {'12': 0, '20': 1, '21': 1, '32': 2, '56': 3, '64': 4, '76': 5}

        self.fft_length = 24

        self.datasets_files_names = {'fft_inplace': 'fft_inplace.csv', 'fft_walk': 'fft_walk.csv'}

        self.extracted_steps_file_name = 'extracted_steps'
        self.extracted_cycles_file_name = 'extracted_cycles'

        self.validated_steps_file_name = 'validated_steps'
        self.validated_cycles_file_name = 'validated_cycles'

        self.step_padding_size = 150
        self.cycle_padding_size = 500

        self.dwt_wavelet = 'db4'
        self.dwt_level = 2
        self.dwt_mode = 'zero'
