#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.save_util import save_to_pkl
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.sac import SAC

import gym
from pathlib import Path
import numpy as np
import json, yaml
from multiprocessing import Pool
import sys
from itertools import starmap
from functools import partial
import gym_grabbing
from gym_grabbing.envs.utils import MLP
import torch as th

# bad!
import sys, os
root = Path(__file__).parent.parent
sys.path.append(str(root/'src'))
from controllers import InterpolateKeyPointsEndPauseGripAssumption, InterpolateKeyPointsGrip

from sbil.utils import TimeLimitAware, scale_action

FOLDER = "/Users/Yakumo/Downloads/exp2/kukaVel2"#"/Users/Yakumo/Downloads/kukaPdStable/kukaSingleConfPdStable/run10"
with open(next(Path(FOLDER).glob('**/run_details.yaml')), 'r') as f:
	INFO = yaml.safe_load(f) # get some common parameters
#INFO['mode'] = 'joint torques' # set it if pd stable to joint torques
#ENV = gym.make(f"{INFO['env id']}", display=False, obj=INFO['object'] ,steps_to_roll=INFO['steps to roll'], mode=INFO['mode'])
#INFO['env kwargs']['display']=True
ENV = gym.make(**INFO['env kwargs'])
time_limit_aware = True
if time_limit_aware:
	ENV = TimeLimitAware(ENV, max_episode_steps=INFO['controller info']['n_iter'])

def simulate(ind, object_position=None, object_xyzw=None, joint_positions=None, position2torque=False, success_only=False, fast_only=float('inf'), noise=0): # return a list of transitions (s, s', a, r, done) if there is a grasping else None

	assert fast_only > 0, "must be positive"
	global ENV
	l, u, = ENV.lowerLimits, ENV.upperLimits
	controller_info = INFO['controller info']
	o = previous_observation = ENV.reset(object_position=object_position, object_xyzw=object_xyzw, joint_positions=joint_positions)
	controller = InterpolateKeyPointsGrip(ind, **controller_info, initial=ENV.get_joint_state(), a_min=-1, a_max=1)
	transitions = []
	achieved = False
	time_before_success = 0
	for k in range(controller_info['n_iter']): # simulation
		action = controller.get_action(k,o)
		if position2torque: # convert position to torque
			#assert INFO['mode'] == 'joint torques', "When using position2torque, INFO['mode'] must be 'joint torques'"
			action[:-1] = ENV.pd_controller.computePD(
				bodyUniqueId=ENV.robot_id,
				jointIndices=ENV.joint_ids,
				desiredPositions=l+(np.hstack((action[:-1], ENV.get_fingers(action[-1])))+1)/2*(u-l),
				desiredVelocities=np.zeros(ENV.n_joints),
				kps=ENV.kps,
				kds=ENV.kds,
				maxForces=ENV.maxForce,
				timeStep=ENV.time_step
			)[:-ENV.n_control_gripper] / ENV.maxForce[:-ENV.n_control_gripper]
		action_w_noise = np.array(action)
		action_w_noise[:-1] += np.random.normal(0,noise, size=(ENV.n_actions-1,))

		if np.logical_or(action>1, action<-1).any():
			return None # invalid action
		o, r, done, inf = ENV.step(action_w_noise)

		if r and not achieved:
			achieved = True
		elif achieved and not r:
			if success_only: transitions = [] # grasp then drop: discard before
			achieved = False

		if achieved or not success_only:
			transitions.append([previous_observation, o, action, r, done, inf])
		previous_observation = o
		time_before_success += not achieved
	if success_only:
			# put the robot into a reasonable position to get good examples
			# works only with otor control, not torque control
			ENV.env.mode = "inverse kinematics"
			for i in range(500):
				o, r, done, info = ENV.step([0,-0.2,0, 1,0,0,0, -1])
				if r:
					transitions.append([previous_observation, o, action, r, done, inf])
					previous_observation = o
				else:
					ENV.env.mode = INFO["env kwargs"]["mode"]
					return None
			ENV.env.mode = INFO["env kwargs"]["mode"]

	if r and time_before_success<fast_only:
		transitions[-1][4] = True # done
		transitions[-1][5]['TimeLimit.truncated'] = True
		return transitions
	else: return None

def generateBuffer(bufferSize=1000000, reward_on=False, success_only=True, npmp_encoder=None, noise=0.05):
	"""save a replay buffer for one object. If reward_on is set to True, all the rewards are set 1."""
	global ENV

	inPath = Path(FOLDER)
	other_individuals, individuals = [], []
	for run_details in inPath.glob('**/run_details.yaml'):
		with open(run_details, 'r') as f:
			d = yaml.safe_load(f)
		if not d['successful']: continue
		data = d["initial state"]['object_position'], d["initial state"]['object_xyzw'], d["initial state"]['joint_positions'], d['env kwargs']['mode']=='pd stable'
		#data = d['object_position'], d['object_xyzw'], d['joint_positions'], d['mode']=='pd stable'
		individuals += [(np.load(f), *data) for f in run_details.parent.glob('type*.npy')]
		other_individuals += [(ind, *data) for ind in np.load(run_details.parent/"individuals.npz")["genotypes"]]
	if len(individuals) == 0: sys.exit("no individual found")
	#individuals = individuals[:10]

	if npmp_encoder is not None:
		npmp_encoder_ = MLP.load(npmp_encoder)
		K = int(npmp_encoder_.observation_space.shape[0]/ENV.robot_space.shape[0])
		latent_dim = int(npmp_encoder_.action_space.shape[0]/2)
		distribution = DiagGaussianDistribution(get_action_dim(npmp_encoder_.action_space))
	else: K = 0

	with Pool() as p:
		episodes = tuple(filter(None, p.starmap(partial(simulate, success_only=success_only, fast_only=float('inf'), noise=noise), individuals)))

	if len(episodes) == 0: sys.exit("all individuals failed")
	#transitions_array = np.vstack(np.array(transitions, dtype=object))
	print("success", len(episodes), "eval", len(individuals))

	if bufferSize<0:
		bufferSize = sum([len(episode)-K for episode in episodes])
	replayBuffer = ReplayBuffer(
		buffer_size=bufferSize,
		observation_space=ENV.observation_space,
		action_space=ENV.action_space if npmp_encoder is None else gym.spaces.Box(low=-np.inf, high=np.inf, shape=(latent_dim,)),
		optimize_memory_usage=True
	)

	def add(episodes):
		for episode in episodes:
			robot_states = np.array([t[5]['robot state'] for t in episode])
			for j,t in enumerate(episode[:-K] if K>0 else episode):
				state, nextState, action, reward, done, info = t
				if npmp_encoder is not None:
					meanlatent_logstd = npmp_encoder_(th.as_tensor(robot_states[j+1:j+K+1].flatten(), dtype=th.float))
					mean_latent, log_std = th.split(meanlatent_logstd, int(npmp_encoder_.action_space.shape[0]/2), dim=-1)
					action = distribution.actions_from_params(mean_latent, log_std, deterministic=True).detach().numpy() # deterministic
				action = scale_action(action, replayBuffer.action_space)
				replayBuffer.add(state, nextState, action, True if reward_on else reward, True if j==len(episode)-K-1 else done, [info])
				if replayBuffer.full: return

	add(episodes) # add once
	print("rb", replayBuffer.full, replayBuffer.pos)
	episode_length = np.mean([len(e)-K for e in episodes])

	# fill while not full
	while not replayBuffer.full:
		n_evals = int(2 * (bufferSize-replayBuffer.pos) / episode_length)
		with Pool() as p:
			episodes = tuple(filter(None, p.starmap(partial(simulate, success_only=success_only, fast_only=float('inf'), noise=noise), [other_individuals[i] for i in np.random.choice(len(other_individuals), n_evals, replace=True)])))
		print("success", len(episodes), "eval", n_evals)
		add(episodes)

	#path = Path(__file__).resolve().parent/f"data/replay_buffer_reward_{'on' if reward_on else 'off'}_{INFO['object']}_{INFO['robot']}{'_success_only' if success_only else ''}{'_npmp_encoded' if npmp_encoder is not None else ''}{'_time_limit_aware' if time_limit_aware else ''}"
	path = Path(__file__).resolve().parent/f"data/demo_{INFO['env kwargs']['obj']}_{INFO['robot']}_{INFO['env kwargs']['mode'].replace(' ', '_')}{'_success_only' if success_only else ''}{'_npmp_encoded' if npmp_encoder is not None else ''}{'_time_limit_aware' if time_limit_aware else ''}"
	save_to_pkl(path=path, obj=replayBuffer)
	print(f"Saved as {path}.pkl")


if __name__ == "__main__":
	generateBuffer(bufferSize=1000000, reward_on=False, success_only=False, noise=0)#, npmp_encoder="/Users/Yakumo/Downloads/npmp_encoder")
