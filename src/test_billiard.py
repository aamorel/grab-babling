import gym
import gym_billiard
import numpy as np
import time
env = gym.make('Billiard-v0')

env.reset()
for i in range(1000):
    # Select the action right
    action = np.random.rand(2) * 2 - 1

    # Take a step in the environment and store it in appropriate variables
    obs, reward, done, info = env.step(action)

    # Render the current state of the environment
    env.render()

    time.sleep(0.01)
