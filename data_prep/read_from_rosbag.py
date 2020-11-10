import rosbag
import pathlib
import re
import os
import pandas as pd

from settings import Settings
from data_prep.general_functions import check_directory_accessibility, check_and_create_directory, save_dataframe_as_csv
from data_prep.read_rosbag import translate_leg_name_and_add_offset


def read_from_rosbag(rosbag_dir_path, raw_haptic_data_dir_path, legs_names, translated_leg_names_dict,
                     f_t_offsets_dict):
    # check if rosbags path is available
    check_directory_accessibility(rosbag_dir_path, 'ROSbags directory not accessible')

    # check if path of save target directory is available, if not create directory
    check_and_create_directory(raw_haptic_data_dir_path,
                               'Problem with creating raw haptic data directory')

    # iterate through all rosbags
    for root, dirs, files in os.walk(rosbag_dir_path):
        for name in files:
            if pathlib.Path(name).suffix == '.bag':

                # load rosbag
                bag_path = os.path.join(root, name)
                bag = rosbag.Bag(bag_path)

                # get list of topics from rosbag
                list_of_topics = bag.get_type_and_topic_info().topics

                # crate dict to store lists of rows of haptic data of specific leg
                rows_legs_dict = {}
                for leg_name in legs_names:
                    rows_legs_dict[leg_name] = []

                # filter through messages of topics of interest (topics that ends with 'wrench' containing needed data)
                for topic, msg, t in bag.read_messages(topics=[x for x in list_of_topics if re.findall("wrench$", x)]):
                    # get leg name from topic
                    leg_name = topic.split('/')[2]

                    # check if translating and adding offset is necessary
                    leg_name, row_dict = translate_leg_name_and_add_offset(legs_names, translated_leg_names_dict,
                                                                           f_t_offsets_dict, leg_name, msg, t)

                    # add single row of data of specific leg to dict of storing lists
                    rows_legs_dict[leg_name].append(row_dict)



                # save all data from specific leg to pandas DataFrame and into .csv file
                for leg_name in legs_names:
                    # prepare saving file path
                    csv_file_name = os.path.splitext(os.path.basename(bag_path))[0] + '_' + leg_name + '.csv'

                    # create panda Dataframe and save it
                    test_leg_dataframe = pd.DataFrame(rows_legs_dict[leg_name])
                    save_dataframe_as_csv(test_leg_dataframe,
                                          os.path.join(raw_haptic_data_dir_path, csv_file_name))

                bag.close()


if __name__ == '__main__':
    s = Settings()
    read_from_rosbag(s.rosbag_dir_path, s.raw_haptic_data_dir_path, s.legs_names, s.leg_names_translate,
                     s.force_torque_offsets)
