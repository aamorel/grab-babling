import gym
env = gym.make('gym_baxter_grabbing:pepper_grasping-v0', display=False)
env.set_steps_to_roll(2)
n_iter = 3000

action = env.action_space.sample()
for i in range(n_iter):
    o, r, eo, inf = env.step(action)
    action = env.action_space.sample()
    if i % 300 == 0:
        print(o[2])