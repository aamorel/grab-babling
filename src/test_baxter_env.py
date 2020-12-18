import gym
import controllers
import numpy as np
baxter_env = gym.make('gym_baxter_grabbing:baxter_grabbing-v1', display=True)
baxter_env.set_steps_to_roll(10)

controller_info = {'pause_frac': 0.66, 'n_iter': 600, 'ADDITIONAL_GENES': 1}
individual = np.random.rand(344) * 2 - 1
controller = controllers.ClosedLoopEndPauseGripAssumption(individual, controller_info)
action = controller.initial_action
for i in range(600):
    o, r, eo, inf = baxter_env.step(action)
    action = controller.get_action(i, o)
    if i % 30 == 0:
        print(o[2])
        

baxter_env.reset()
print('reset made')
action = controller.initial_action
for i in range(600):
    o, r, eo, inf = baxter_env.step(action)
    action = controller.get_action(i, o)
    if i % 30 == 0:
        print(o[2])
        

baxter_env.reset()
print('reset made')
action = controller.initial_action
for i in range(600):
    o, r, eo, inf = baxter_env.step(action)
    action = controller.get_action(i, o)
    if i % 30 == 0:
        print(o[2])
        
