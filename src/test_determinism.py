import gym
import controllers
import numpy as np
from scoop import futures


n_iter = 1500

controller_info = {'pause_frac': 0.66, 'n_iter': n_iter,
                   'NB_KEYPOINTS': 3, 'GENE_PER_KEYPOINTS': 9}

baxter_env = gym.make('gym_baxter_grabbing:kuka_grasping-v0', display=False, obj='cup')
baxter_env.set_steps_to_roll(1)
n_exp = 100


def eval_func(individual):
    baxter_env.reset()
    controller = controllers.InterpolateKeyPointsEndPauseGripAssumption(individual, controller_info)
    action = controller.initial_action
    obs = []
    for i in range(n_iter):
        o, r, eo, inf = baxter_env.step(action)
        action = controller.get_action(i)
        flat_list = [item for sublist in o for item in sublist]
        obs.append(flat_list)
    return np.array(obs).flatten()


individual = np.random.rand(25) * 2 - 1
individual_list = [individual] * n_exp


if __name__ == '__main__':
    observations = list(map(eval_func, individual_list))

    for obs in observations:
        assert((obs == observations[0]).all())

    # observations = list(futures.map(eval_func, individual_list))

    # for obs in observations:
    #     assert((obs == observations[0]).all())
