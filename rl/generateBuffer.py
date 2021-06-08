#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.save_util import save_to_pkl
from stable_baselines3.sac import SAC
import gym
from pathlib import Path
import numpy as np
import json, yaml
from multiprocessing import Pool
import sys
from itertools import starmap
import gym_baxter_grabbing

# bad!
import sys, os
root = Path(__file__).parent.parent
sys.path.append(str(root/'src'))
from controllers import InterpolateKeyPointsEndPauseGripAssumption

FOLDER = root/"runs"
with open(next(Path(FOLDER).glob('**/run_details.yaml')), 'r') as f:
	INFO = yaml.safe_load(f) # get some common parameters
ENV = gym.make(f"{INFO['robot']}_grasping-v0", display=False, obj=INFO['object'] ,steps_to_roll=INFO['steps to roll'], mode=INFO['mode'])

def simulate(ind, object_position=None, object_xyzw=None, joint_positions=None): # return a list of transitions (s, s', a, r, done) if there is a grasping else None
	#print(object_position)
	global ENV
	#controller_info = {'pause_frac': 0.66, 'n_iter': 600, 'NB_KEYPOINTS': 3, 'GENE_PER_KEYPOINTS': 8}
	controller_info = INFO['controller info']
	previous_observation = ENV.reset(object_position=object_position, object_xyzw=object_xyzw, joint_positions=joint_positions)
	controller = InterpolateKeyPointsEndPauseGripAssumption(ind, controller_info, initial=ENV.get_joint_state())
	action = controller.initial_action
	transitions = [None]*controller_info['n_iter']
	for k in range(controller_info['n_iter']): # simulation
		o, r, eo, inf = ENV.step(action)
		#torque_action = inf['applied joint motor torques'] / ENV.maxForce
		#transitions[k] = previous_observation, o, np.hstack([torque_action[:-ENV.n_control_gripper], action[-1]]), r, eo
		transitions[k] = previous_observation, o, action, r, eo
		previous_observation = o
		action = controller.get_action(k)
	return transitions if r is True else None

def generateBuffer(outFolder, bufferSize=1000000, reward_on=False):
	"""save a replay buffer for one object. If reward_on is set to True, all the rewards are set 1."""
	inPath = Path(FOLDER)
	individuals = []
	for run_details in inPath.glob('**/run_details.yaml'):
		with open(run_details, 'r') as f:
			d = yaml.safe_load(f)
		individuals += [(np.load(f),d['object_position'], d['object_xyzw'], d['joint_positions']) for f in run_details.parent.glob('*type*.npy')]
	if len(individuals) == 0: sys.exit("no individual found")
	

	with Pool() as p:
		transitions = tuple(filter(None, p.starmap(simulate, individuals)))
	#transitions = tuple(filter(None, starmap(simulate, individuals)))
	if len(transitions) == 0: sys.exit("all individuals failed")
	transitions_array = np.vstack(np.array(transitions, dtype=object))
	print('shape',transitions_array.shape, "n_success", len(transitions), "n_individuals", len(individuals))
	
	if bufferSize<0:
		bufferSize = len(transitions_array)
	replayBuffer = ReplayBuffer(buffer_size=bufferSize, observation_space=ENV.observation_space, action_space=ENV.action_space)
	for i in range(np.ceil(bufferSize/len(transitions_array)).astype(int)): # repeat transitions to fill the entire buffer
		for t in transitions_array:
			state, nextState, action, reward, done = t
			replayBuffer.add(state, nextState, action, True if reward_on else reward, done)
	path = outFolder+f"/replayBuffer_{INFO['object']}"
	save_to_pkl(path=path, obj=replayBuffer)
	print(f"Saved as {path}.pkl")
	
def generateExamples(outFolder):
	"""save successfull examples (state only) for one object"""
	inPath = Path(FOLDER)
	individuals = []
	for run_details in inPath.glob('**/run_details.yaml'):
		with open(run_details, 'r') as f:
			d = yaml.safe_load(f)
		#if d['object'] == OBJECT:
		individuals += [(np.load(f),d['object_position'], d['object_xyzw'], d['joint_positions']) for f in run_details.parent.glob('*type*.npy')]
	if len(individuals) == 0: sys.exit("no individual found")
	
	with Pool() as p:
		transitions = tuple(filter(None, p.starmap(simulate, individuals)))
	if len(transitions) == 0: sys.exit("all individuals failed")
	else: print("n_success", len(transitions), "n_individuals", len(individuals))

	examples = np.zeros((2*len(transitions), ENV.observation_space.shape[0])) # save 2 examples per episode
	nextExamples = np.zeros((2*len(transitions), ENV.observation_space.shape[0]))
	actions = np.zeros((2*len(transitions), ENV.action_space.shape[0]))
	for i, episode in enumerate(transitions):
		for state, nextState, action, reward, done in episode:
			if reward: break
		#print(state, episode[-1][0])
		lastState, lastNextState, lastAction, lastReward, lastDone = episode[-2]
		examples[2*i:2*(i+1)] = state, lastState # add first reward signal and last state of the episode
		nextExamples[2*i:2*(i+1)] = nextState, lastNextState
		actions[2*i:2*(i+1)] = action, lastAction
	
	np.savez_compressed(Path(outFolder)/f"examples_{INFO['object']}.npz", **{'examples':examples, 'example_next_states': nextExamples, 'example_actions':actions})

if __name__ == "__main__":
	#SAC('MlpPolicy', ENV).learn(1000)
	#generateBuffer('/Users/Yakumo/Downloads', bufferSize=-1, reward_on=False)
	generateExamples('/Users/Yakumo/Downloads')