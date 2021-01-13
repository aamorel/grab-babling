
import gym_minigrid
import gym
from gym_minigrid.wrappers import ImgObsWrapper, OneHotPartialObsWrapper
import time
import random
env = OneHotPartialObsWrapper(gym.make('MiniGrid-Empty-8x8-v0'))

# Reset the environment
env.reset()

for i in range(1000):
    # Select the action right
    action = random.randint(0, 2)

    # Take a step in the environment and store it in appropriate variables
    obs, reward, done, info = env.step(action)

    o = obs['image'][3, 4:7, 0:6]
    # Render the current state of the environment
    env.render()

    time.sleep(0.01)

