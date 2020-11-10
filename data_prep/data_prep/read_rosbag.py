def encode_rostime(rostime):
    # function that converts rostime into seconds
    try:
        return rostime.to_sec()
    except AttributeError:
        raise TypeError('Received object is not a Time instance.')


def add_offset(ros_message_converted_to_dict, offset_dict, leg_name):
    # adds necessary offset to collected data
    for offset_name in offset_dict[leg_name]:
        ros_message_converted_to_dict[offset_name] += offset_dict[leg_name][offset_name]
    return ros_message_converted_to_dict


def convert_ros_message_to_dict(ros_message, rostime):
    return {'time': encode_rostime(rostime), 'force_x': ros_message.wrench.force.x,
            'force_y': ros_message.wrench.force.y,
            'force_z': ros_message.wrench.force.z, 'torque_x': ros_message.wrench.torque.x,
            'torque_y': ros_message.wrench.torque.y,
            'torque_z': ros_message.wrench.torque.z}


def translate_leg_name_and_add_offset(legs_names, leg_names_translate, force_torque_offsets, leg_name, ros_message,
                                      rostime):
    # simplify two different ways off naming legs used in
    # data collecting process and adds necessary offset to collected data
    ros_message_converted_to_dict = convert_ros_message_to_dict(ros_message, rostime)
    # check if translation is needed
    if leg_name not in legs_names:
        # if needed translate name and add offset
        leg_name = leg_names_translate[leg_name]
        ros_message_converted_to_dict = add_offset(ros_message_converted_to_dict, force_torque_offsets, leg_name)

    return leg_name, ros_message_converted_to_dict
