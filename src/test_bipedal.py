import gym
import numpy as np
import torch
import torch.nn as nn


class ControllerNetBipedal(nn.Module):

    def __init__(self, params):
        super(ControllerNetBipedal, self).__init__()
        self.l1 = nn.Linear(14, 6)
        weight_1 = nn.Parameter(torch.Tensor(params[:84]).reshape((6, 14)))
        bias_1 = nn.Parameter(torch.Tensor(params[84:90]).reshape(6))
        self.l1.weight = weight_1
        self.l1.bias = bias_1
        self.l2 = nn.Linear(6, 4)
        weight_2 = nn.Parameter(torch.Tensor(params[90:114]).reshape((4, 6)))
        bias_2 = nn.Parameter(torch.Tensor(params[114:118]).reshape(4))
        self.l2.weight = weight_2
        self.l2.bias = bias_2
        self.r1 = nn.ReLU()
        self.r2 = nn.Tanh()

    def forward(self, obs):
        obs = torch.Tensor(obs)
        obs = self.r1(self.l1(obs))
        action = self.r2(self.l2(obs)).numpy()

        return action


env = gym.make('BipedalWalker-v3')
env.reset()
action = [0, 0, 0, 0]
ind = np.random.rand(118) * 2 - 1
controller = ControllerNetBipedal(ind)
while True:
    s, r, done, info = env.step(action)
    s = s[:14]
    with torch.no_grad():
        action = controller(s)
    env.render()
    print(env.hull.position)
    if done:
        env.reset()
        action = [0, 0, 0, 0]
        ind = np.random.rand(118) * 2 - 1
        controller = ControllerNetBipedal(ind)
