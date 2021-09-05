# plot trajectories of the end-effector in pybullet

from multiprocessing import Pool
import time
from pathlib import Path
from functools import partial

import gym
import numpy as np
import json, yaml
import gym_grabbing
from gym_grabbing.envs.utils import InterpolateKeyPointsGrip

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--runs", help="The directory containing runs", type=str, default=str(Path(__file__).parent.parent/'runs'))
args = parser.parse_args()

FOLDER = args.runs
with open(next(Path(FOLDER).glob('**/run_details.yaml')), 'r') as f:
    INFO = yaml.safe_load(f) # get some common parameters
#ENV = gym.make(f"{INFO['env id']}", display=False, obj=INFO['object'] ,steps_to_roll=INFO['steps to roll'], mode=INFO['mode'])
ENV = gym.make(**INFO['env kwargs'])

def simulate(data, initial_state={}, period=1): # return a list of transitions (s, s', a, r, done) if there is a grasping else None
    ind, initial_state = data
    global ENV
    controller_info = INFO['controller info']
    o = previous_observation = ENV.reset(**initial_state)
    controller = InterpolateKeyPointsGrip(ind, **controller_info, initial=ENV.get_joint_state(), a_min=-1, a_max=1)
    trajectory = np.zeros((np.ceil(controller_info['n_iter']/period).astype(int), 3))
    trajectory[0] = ENV.info['end effector position']
    for i in range(controller_info['n_iter']):
        o, r, done, info = ENV.step(controller.get_action(i,o))
        if i>0 and i%period==0:
            trajectory[i//period] = info['end effector position']
    return trajectory

def plot_trajectories(period=4):
    inPath = Path(FOLDER)
    all_individuals, individuals = [], []
    for run_details in inPath.glob('**/run_details.yaml'):
        with open(run_details, 'r') as f:
            d = yaml.safe_load(f)
        if not d['successful']: continue
        individuals += [(np.load(f), d["initial state"]) for f in run_details.parent.glob('type*.npy')]
        all_individuals_run = np.load(run_details.parent/"individuals.npz")["genotypes"][:1000]
        np.random.shuffle(all_individuals_run)
        all_individuals += [(ind, d["initial state"]) for ind in all_individuals_run]
    if len(individuals) == 0:
        print("no individual found")
        return

    with Pool() as p:
        trajectories = np.array(p.map(partial(simulate, period=period), all_individuals))

    INFO['env kwargs']['display'] = True
    env = gym.make(**INFO['env kwargs'])
    env.reset()
    for j, trajectory in enumerate(trajectories):
        last_pos = trajectory[0]
        for i, position in enumerate(trajectory[1:]):
            k = i/(INFO['controller info']["n_iter"]//period)
            env.p.addUserDebugLine(last_pos, position, [0, k, 1-k], parentObjectUniqueId=-1, lineWidth=10)
            last_pos = position


    input("press enter to continue")

if __name__ == '__main__':
    plot_trajectories()
