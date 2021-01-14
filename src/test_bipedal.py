import gym
import numpy as np
import utils


env = gym.make('BipedalWalker-v3')
env.reset()
action = [0, 0, 0, 0]
ind = np.random.rand(118) * 2 - 1
controller = utils.NeuralAgentNumpy(14, 4, n_hidden_layers=1, n_neurons_per_hidden=6)
controller.set_weights(ind)
while True:
    s, r, done, info = env.step(action)
    s = s[:14]
    action = controller.choose_action(s)
    env.render()
    print(env.hull.position)
    if done:
        env.reset()
        action = [0, 0, 0, 0]
        ind = np.random.rand(118) * 2 - 1
        controller = utils.NeuralAgentNumpy(14, 4, n_hidden_layers=1, n_neurons_per_hidden=6)
        controller.set_weights(ind)
