#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.save_util import save_to_pkl, load_from_pkl
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC, PPO, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, ConvertCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit, ReplayBufferSamples
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common.logger import Logger, CSVOutputFormat, configure, Video
#from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common import results_plotter
import gym
import gym_grabbing
import numpy as np
from algoRL import TQC_RCE, TQC_SQIL, TQC_RED, SAC_RCE, InverseModel, ForwardModel, train_dynamic_model, initialize_bc_policy, behaviouralCloningWithModel
from sb3_contrib import TQC
from functools import partial
from pathlib import Path
import os
import yaml
import imageio
import torch as th
th.autograd.set_detect_anomaly(True)

from typing import Union, Optional

class Callback(EvalCallback):
	def __init__(self, *args, **kwargs):
		super(Callback, self).__init__(*args, **kwargs)
		self.count = 0
		
	#def _on_training_start(self): # setup the csv logger
		#self.dir = self.logger.get_dir() or self.log_path
		#self.logger.reset() # reconfigure to add csv
		#configure(folder=self.dir, format_strings=['csv', 'tensorboard'] if self.model.tensorboard_log is not None else ['csv'])
		
	def _log_success_callback(self, locals_, globals_) -> None:
		super()._log_success_callback(locals_, globals_)
		if self.count%40 == 0: # get 24 images per second because the fps in pybullet is 240 / rollout
			self.images.append(self.eval_env.render(mode='rgb_array')) #.transpose(2, 0, 1)
		self.count += 1
		
		
	def _on_step(self) -> bool:

		if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
			self.count = 0
			self.images = []
				
			out = super()._on_step()
			#self.logger.record(
			#"trajectory/video",
				#Video(th.ByteTensor([self.images]), fps=24),
				#exclude=("stdout", "log", "json", "csv"),
			#)
			#print(len(self.images), self.images[0].shape)
			imageio.mimsave(self.dir+"/eval.gif", self.images, fps=6)
			self.model.save(self.dir+"/last_model.zip")
			return out
		return True
		
	

def learnReach(log_path, vec_env=False, mode='joint torques', her=False):
	env_kwargs = dict(id='kuka_grasping-v0', display=False, obj='cube', steps_to_roll=1, mode=mode, reset_random_initial_state=True, reach=True, goal=her)
	max_episode_steps = 2000
	env = lambda : gym.wrappers.TimeLimit(gym.make(**env_kwargs), max_episode_steps=max_episode_steps)#'kuka_grasping-v0', display=display, obj='cube', steps_to_roll=1, mode='joint torques', reset_random_initial_state=True, reach=True)
	
	eval_callback = Callback(env(), best_model_save_path=log_path, log_path=log_path, eval_freq=25000, deterministic=True, render=False, n_eval_episodes=5)
	interval = 64
	
	if her:
		policy = "MultiInputPolicy"
		replay_buffer_class = HerReplayBuffer
		replay_buffer_kwargs = { # Parameters for HER
			"n_sampled_goal": 4,
			"goal_selection_strategy": "future",
			"online_sampling": True,
			#"max_episode_length": max_episode_length, # can be inferred with gym.wrappers.TimeLimit
		}
	else:
		policy = 'MlpPolicy'
		replay_buffer_class, replay_buffer_kwargs = None, None
		
	model = TQC(
		policy=policy,
		env=make_vec_env(env_id=env_kwargs.pop('id'), n_envs=os.cpu_count(), env_kwargs=env_kwargs) if vec_env else env(),
		learning_rate=0.0007,
		tensorboard_log=log_path,
		learning_starts=max_episode_steps, # we must initialize the replay buffer with at least one episode to make HerReplayBuffer work
		tau=0.02,
		train_freq=interval,
		gradient_steps=interval,
		target_update_interval=1,
		policy_kwargs={'net_arch':dict(qf=[400, 300], pi=[256, 256]), 'activation_fn':th.nn.LeakyReLU},
		device='cpu',
		replay_buffer_class=replay_buffer_class,
		replay_buffer_kwargs=replay_buffer_kwargs,
	)
	model.set_logger(configure(folder=str(log_path), format_strings=["stdout", "csv", "tensorboard"])) # save csv as well
	model.learn(10000000, callback=eval_callback, tb_log_name='TQC_reach')
	
	
def learnSimple(log_path, ReplayBufferPath=None, sqil=False, action_strategy='inverse model', device='auto'):
	""" ReplayBufferPath is the path of the pickled ReplayBuffer
	if sqil is set to True, all rewards must be set to 1 in the replay buffer
	otherwise, it is a simple RL algorithm (TQC) with a ReplayBuffer initialized if given
	"""
	env = lambda display=False: gym.make('kuka_grasping-v0', display=display, obj='cube', steps_to_roll=1, mode='joint torques')#, reset_random_initial_state=True, reach=True)
	# Use deterministic actions for evaluation
	eval_callback = Callback(env(), best_model_save_path=log_path, log_path=log_path, eval_freq=25000, deterministic=False, render=False, n_eval_episodes=2)

	model_class = partial(
		TQC_SQIL if sqil else TQC,
		policy='MlpPolicy',
		env=env(),
		tensorboard_log=log_path,
		learning_starts=20000,
		device=device,
		use_sde=False,
		policy_kwargs={'net_arch':dict(qf=[400,300], pi=[256, 256]), 'activation_fn':th.nn.LeakyReLU},
		target_update_interval=1,
	) #ent_coef=1e-3)#, train_freq=1, batch_size=512)
	if sqil:
		assert isinstance('ReplayBufferPath', str), 'ReplayBufferPath must be provided if using SQIL'
		model = model_class(expert_replay_buffer=ReplayBufferPath, action_strategy=action_strategy)
		model, inverseModel = behaviouralCloningWithModel(collection_timesteps=10000, use_inverse=True, repeat=5, model=model, expert_replay_buffer=ReplayBufferPath, device=device)
	else:
		model = model_class()
		if ReplayBufferPath is not None:
			model.load_replay_buffer(ReplayBufferPath)
	
	model.learn(10000000, callback=eval_callback, tb_log_name='sqil' if sqil else 'TQC')
	
def learnRCE(log_path, examplesPath, pretrain=None, device='auto', action_strategy='inverse model'):
	env = lambda: gym.make('kuka_grasping-v0', display=False, obj='cube', steps_to_roll=1, mode='joint torques')#, reset_random_initial_state=True)
	env = DummyVecEnv([env])
	
	# Use deterministic actions for evaluation
	eval_callback = Callback(env, best_model_save_path=log_path, log_path=log_path, eval_freq=25000, deterministic=False, render=False, n_eval_episodes=2)
	
	#states, next_states, actions = np.load(examplesPath).values()
	
	model = TQC_RCE(
		**np.load(examplesPath),
		policy='MlpPolicy',
		env=env,
		learning_starts=200000,
		tensorboard_log=log_path,
		device=device,
		action_strategy=action_strategy,
		use_sde=False,
		policy_kwargs={'net_arch':dict(qf=[256, 256], pi=[256, 256]), 'activation_fn':th.nn.LeakyReLU},
		target_update_interval=1,
	) #, batch_size=1024, train_freq=10, policy_kwargs={'net_arch':dict(pi=[256, 128], qf=[512, 256])}, target_update_interval=10
	if pretrain is not None:
		model, inverseModel = behaviouralCloningWithModel(collection_timesteps=10000, use_inverse=True, repeat=5, model=model, expert_replay_buffer=pretrain, device=device)
	
	model.learn(10000000, callback=eval_callback, tb_log_name='RCE')
	
def learnRED(log_path, demonstration_replay_buffer, action_strategy='current policy', device='auto'):
	env = lambda display=False: gym.make('kuka_grasping-v0', display=display, obj='cube', steps_to_roll=1, mode='joint torques')#, reset_random_initial_state=True, reach=True)
	eval_callback = Callback(env(), best_model_save_path=log_path, log_path=log_path, eval_freq=25000, deterministic=False, render=False, n_eval_episodes=2)
	model = TQC_RED(
		demonstration_replay_buffer=demonstration_replay_buffer,
		policy='MlpPolicy',
		env=env(),
		learning_starts=200000,
		tensorboard_log=log_path,
		device=device,
		action_strategy=action_strategy,
		use_actions=False,
		use_sde=False,
		policy_kwargs={'net_arch':dict(qf=[400, 300], pi=[256, 256]), 'activation_fn':th.nn.LeakyReLU},
		target_update_interval=1,
	)
	#model.pretrain(collection_timesteps=10000, repeat=2, expert_replay_buffer=demonstration_replay_buffer)
	model.learn(total_timesteps=10000000, callback=eval_callback, tb_log_name='RED')

	
def testBC2(): # raw bc
	env = lambda display=False: gym.make('kuka_grasping-v0', display=False, obj='cube', steps_to_roll=1, mode='joint torques', reset_random_initial_state=True)
	model = TQC(policy='MlpPolicy', env=env(), learning_starts=1000000, gradient_steps=1, batch_size=512)
	policy_loss = initialize_bc_policy(model, expert_replay_buffer=load_from_pkl('/Users/Yakumo/Downloads/replayBuffer_cubeTorque.pkl', model.verbose), dynamic_model=None, n_iter=100000, batch_size=512)
	print("policy loss", policy_loss)
	model.save('/Users/Yakumo/Downloads/TQC_bc.pkl')
	
def testInverseModel():
	dynamic_model = InverseModel.load('/Users/Yakumo/Downloads/inverseModel.pkl')
	replay_buffer = load_from_pkl('/Users/Yakumo/Downloads/replayBuffer_cubeTorque.pkl')
	replay_data = replay_buffer.sample(batch_size=1)
	print(np.round(replay_data.observations.squeeze()[15:].tolist(),3).tolist())
	print(np.round(replay_data.next_observations.squeeze()[15:].tolist(),3).tolist())
	print(dynamic_model(replay_data.observations, replay_data.next_observations).squeeze().tolist())
	print(replay_data.actions.squeeze().tolist())
	
def enjoy(model_path, Model):
	#with open('/Users/Yakumo/Downloads/kukaNoContactTable/cube__682847[92].mesu2_2021-05-19_21:04:40_run0_kukaPos/run_details.yaml', 'r') as f:
		#d = yaml.safe_load(f)
	#env = gym.make('kuka_grasping-v0', display=True, obj='cube', steps_to_roll=1, mode='joint torques', object_position=d['object_position'], object_xyzw=d['object_xyzw'], joint_positions=d['joint_positions'])#, early_stopping=True)
	env = gym.make('kuka_grasping-v0', display=True, obj='cube', steps_to_roll=1, mode='joint torques', reset_random_initial_state=True)
	model = Model.load(model_path, env=env)
	mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
	print(mean_reward, std_reward)
	
def plot(log_dir): # need Monitor
	plot_results([log_dir], num_timesteps=None, x_axis=results_plotter.X_TIMESTEPS, task_name="TD3 LunarLander")
	plt.show()
	
	
if __name__ == '__main__':
	folder = Path(__file__).resolve().parent
	log_path = folder/"log"
	data_path = folder/"data"
	log_path.mkdir(exist_ok=True)
	
	learnReach(log_path=log_path, vec_env=False, mode='joint torques', her=True)
	#learnSimple(log_path=log_path, ReplayBufferPath='/Users/Yakumo/Downloads/replayBufferCubeSingleConfigRewardOff.pkl', sqil=True, action_strategy='inverse model')
	#learnSimple(log_path=log_path, ReplayBufferPath='/home/yakumo/Documents/AurelienMorel/rl/replayBufferCubeSingleConfigRewardOff.pkl', sqil=True, action_strategy='inverse model')
	#learnRCE(log_path=log_path, '/Users/Yakumo/Downloads/examples_cube.npz', pretrain='/Users/Yakumo/Downloads/replayBufferCubeSingleConfigRewardOff.pkl', action_strategy='inverse model')
	#learnRCE(log_path=log_path, '/home/yakumo/Documents/AurelienMorel/rl/examples_cube.npz', pretrain='/home/yakumo/Documents/AurelienMorel/rl/replayBufferCubeSingleConfigRewardOff.pkl', action_strategy='inverse model')
	#enjoy('/Users/yakumo/Downloads/last_model.zip', TQC)
	#behaviouralCloningWithModel(collection_timesteps=10000, use_inverse=True, repeat=5, expert_replay_buffer='/Users/Yakumo/Downloads/replayBufferCubeSingleConfigRewardOff.pkl')
	#testBC2()
	#plot('/Users/Yakumo/Downloads/TQC')
	#learnRED(log_path=log_path, demonstration_replay_buffer='/Users/Yakumo/Downloads/replayBufferCubeSingleConfigRewardOff.pkl')
	print('end')
	
