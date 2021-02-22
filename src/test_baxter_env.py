import gym
import controllers
import numpy as np

n_iter = 3000
# controller_info = {'pause_frac': 0.66, 'n_iter': n_iter, 'ADDITIONAL_GENES': 1}
# for _ in range(3):
#     baxter_env.reset()
#     individual = np.random.rand(344) * 2 - 1
#     print('new individual closed loop')
#     controller = controllers.ClosedLoopEndPauseGripAssumption(individual, controller_info)
#     action = controller.initial_action
#     for i in range(n_iter):
#         o, r, eo, inf = baxter_env.step(action)
#         action = controller.get_action(i, o)
#         if i % 300 == 0:
#             print(o[2])

controller_info = {'pause_frac': 0.66, 'n_iter': n_iter,
                   'NB_KEYPOINTS': 3, 'GENE_PER_KEYPOINTS': 9}
for _ in range(3):
    baxter_env = gym.make('gym_baxter_grabbing:kuka_grasping-v0', display=False, obj='cup')
    baxter_env.set_steps_to_roll(1)
    individual = np.random.rand(25) * 2 - 1
    print('new individual open loop')
    controller = controllers.InterpolateKeyPointsEndPauseGripAssumption(individual, controller_info)
    action = controller.initial_action
    for i in range(n_iter):
        o, r, eo, inf = baxter_env.step(action)
        action = controller.get_action(i)
        if i % 150 == 0:
            print(inf['closed gripper'])
            # print(o[4][-4:])
    baxter_env.close()
