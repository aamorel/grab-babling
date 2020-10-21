import gym

baxter_env = gym.make('gym_baxter_grabbing:baxter_grabbing-v1', display=True)

for i in range(1000):
    baxter_env.step([0.2, 0, -0.1, 0.2, 0.2, -0.2, -0.2, 1])
