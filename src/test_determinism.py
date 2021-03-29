import gym
import controllers
import numpy as np
import copy


n_iter = 300

controller_info = {'pause_frac': 0.66, 'n_iter': n_iter,
                   'NB_KEYPOINTS': 3, 'GENE_PER_KEYPOINTS': 9}

n_exp = 20
n_ind = 20


def eval_func(individual, baxter_env):
    baxter_env.reset()
    controller = controllers.InterpolateKeyPointsEndPauseGripAssumption(individual, controller_info)
    action = controller.initial_action
    obs = []
    for i in range(n_iter):
        o, r, eo, inf = baxter_env.step(action)
        action = controller.get_action(i)
        flat_list = [item for sublist in o for item in sublist]
        obs.append(flat_list)
#    return np.array(obs).flatten()
    return o[0]


if __name__ == '__main__':
    unique_obs_lens = []
    for i in range(n_ind):
        baxter_env = gym.make('gym_baxter_grabbing:kuka_grasping-v0', display=False, obj='cup', steps_to_roll=1)

        individual = np.random.rand(25) * 2 - 1
        individual_list = [copy.deepcopy(individual)] * n_exp
        observations = []
        for ind in individual_list:
            observations.append(eval_func(ind, baxter_env))
        baxter_env.close()
        unique_obs = []
        observations = np.array(observations)
        uniques = np.unique(observations.round(decimals=6), axis=0)
        unique_obs_lens.append(len(uniques))
        # for obs in observations[1:]:
        #     assert(np.isclose(obs, observations[0]).all())
    print(unique_obs_lens)

    # observations = list(futures.map(eval_func, individual_list))

    # for obs in observations:
    #     assert((obs == observations[0]).all())
