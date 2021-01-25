import gym
import gym_billiard
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Billiard-v0')

print('Actions {}'.format(env.action_space))
print('Obs {}'.format(env.observation_space))

action = env.action_space.sample()
lim = 20
total_act = np.array([0., 0.])
for i in range(1):
    obs = env.reset()
    a = env.render(mode='rgb_array')

    for t in range(10000):
        if t < lim:
            action = [0.5, 1.]
            total_act += np.array(action)

        else:
            action = [0., 0.]
        img = env.render()
        env.render(mode='human')

        plt.imshow(img, interpolation='nearest')
        plt.draw()
        obs, reward, done, info = env.step(action)
        if done:
            break
    print(info)
