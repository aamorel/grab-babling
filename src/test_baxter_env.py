import gym

baxter_env = gym.make('gym_baxter_grabbing:baxter_grabbing-v0', display=True)

for i in range(1000):
    baxter_env.step([0.2, 0, -0.1])
