import math
from scipy import interpolate


class DiscreteKeyPoints():

    def __init__(self, actions, n_iter, additional_genes):
        """Share the time equally between keypoints

        Args:
            actions (list): list of actions
            n_iter (int): max number of iterations
        """
        self.actions = actions
        self.n_iter_per_action = math.ceil(n_iter / len(actions))

    def get_action(self, i):
        action_index = i // self.n_iter_per_action
        action = self.actions[action_index]
        return action
        

class InterpolateKeyPoints():

    def __init__(self, actions, n_iter, additional_genes):
        """Interpolate actions between keypoints

        Args:
            actions (list): list of actions
            n_iter (int): max number of iterations
        """
        n_keypoints = len(actions)
        interval_size = int(n_iter / n_keypoints)
        interp_x = [int(interval_size / 2 + i * interval_size) for i in range(n_keypoints)]
        self.action_polynome = interpolate.interp1d(interp_x, actions, kind='quadratic', axis=0,
                                                    bounds_error=False, fill_value='extrapolate')

    def get_action(self, i):
        action = self.action_polynome(i)
        return action


class InterpolateKeyPointsEndPause():

    def __init__(self, actions, n_iter, additional_genes, pause_frac=0.66):
        """Interpolate actions between keypoints
           Stops the movement at the end in order to make sure that the object was correctly grasped at the end.

        Args:
            actions (list): list of actions
            n_iter (int): max number of iterations
        """
        n_keypoints = len(actions)
        self.pause_time = pause_frac * n_iter
        interval_size = int(self.pause_time / n_keypoints)
        interp_x = [int(interval_size / 2 + i * interval_size) for i in range(n_keypoints)]
        self.action_polynome = interpolate.interp1d(interp_x, actions, kind='quadratic', axis=0,
                                                    bounds_error=False, fill_value='extrapolate')
        self.last_action = 0

    def get_action(self, i):
        if i <= self.pause_time:
            action = self.action_polynome(i)
            self.last_action = action
        else:
            action = self.last_action
        return action


class InterpolateKeyPointsEndPauseGripAssumption():

    def __init__(self, actions, n_iter, additional_genes, pause_frac=0.66):
        """Interpolate actions between keypoints
           Stops the movement at the end in order to make sure that the object was correctly grasped at the end.
           Only one parameter for the gripper, specifying the time at which it should close

        Args:
            actions (list): list of actions
            n_iter (int): max number of iterations
        """
        n_keypoints = len(actions)
        self.pause_time = pause_frac * n_iter
        interval_size = int(self.pause_time / n_keypoints)
        interp_x = [int(interval_size / 2 + i * interval_size) for i in range(n_keypoints)]
        self.action_polynome = interpolate.interp1d(interp_x, actions, kind='quadratic', axis=0,
                                                    bounds_error=False, fill_value='extrapolate')
        self.last_action = 0
        self.grip_time = math.floor(additional_genes / 2 + 0.5 * self.pause_time)

    def get_action(self, i):
        if i <= self.pause_time:
            action = self.action_polynome(i)
            if i < self.grip_time:
                action.append(1)  # gripper is open
            else:
                action.append(-1)  # gripper is closed
            self.last_action = action
        else:
            # gripper is closed
            action = self.last_action
        return action
