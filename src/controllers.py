import math


def controller_discrete_key_points(i, actions, n_iter, n_keypoints):
    """Share the keypoints time equally

    Args:
        i (int): iteration index
        actions (list): reference for list of actions

    Returns:
        list: action
    """
    nb_iter_per_action = math.ceil(n_iter / n_keypoints)
    action_index = i // nb_iter_per_action
    action = actions[action_index]
    return action
