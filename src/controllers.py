import math
from scipy import interpolate


class ControllerDiscreteKeyPoints():

    def __init__(self, actions, n_iter):
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
        

class ControllerInterpolateKeyPoints():

    def __init__(self, actions, n_iter):
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
