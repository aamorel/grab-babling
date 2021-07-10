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
from algoRL import TQC_RCE, TQC_SQIL, TQC_RED, SAC_RCE, InverseModel, ForwardModel, train_dynamic_model, initialize_bc_policy, behaviouralCloningWithModel, TQC_PWIL
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
		
	def _on_training_start(self): # setup the csv logger
		self.dir = self.logger.get_dir() or self.log_path
		logger = configure(folder=self.dir, format_strings=['csv', 'tensorboard'] if self.model.tensorboard_log is not None else ['csv'])
		self.model.set_logger(logger) # set logger to the model
		self.logger = logger # set logger to the callback
		
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
	model.learn(10000000, callback=eval_callback, tb_log_name='tqc_reach')
	
	


def learn(model_class, kwargs=None, load=None, log_path=None, device='auto', load_replay_buffer=None, max_episode_steps=1500, env_kwargs=None, enjoy=None):
	env_kwargs_ = env_kwargs or dict(id='kuka_grasping-v0', display=False, obj='cube', steps_to_roll=1, mode='joint torques')
	env = lambda display=False: gym.wrappers.TimeLimit(gym.make(**env_kwargs_), max_episode_steps=max_episode_steps)
	#env = lambda: VecCheckNan(DummyVecEnv([env]), raise_exception=True)
	eval_callback = Callback(env(), best_model_save_path=log_path, log_path=log_path, eval_freq=25000, deterministic=True, render=False, n_eval_episodes=2)
	if load is not None:
		model = model_class.load(load)
	else:
		model = model_class(
			policy='MlpPolicy',
			env=env(),
			tensorboard_log=log_path,
			device=device,
			policy_kwargs={'net_arch':dict(qf=[256, 256], pi=[256, 256]), 'activation_fn':th.nn.LeakyReLU},
			**(kwargs or {}),
		)
	
	if load_replay_buffer:
		model.load_replay_buffer(load_replay_buffer)
	
	if enjoy:
		print("reward: mean={}, std={}".format(*evaluate_policy(model, model.get_env(), n_eval_episodes=enjoy, deterministic=True)))
	else:
		model.learn(total_timesteps=10000000, callback=eval_callback, tb_log_name=model_class.__name__.lower())

	
def testBC2(): # raw bc
	env = lambda display=False: gym.make('kuka_grasping-v0', display=False, obj='cube', steps_to_roll=1, mode='joint torques', reset_random_initial_state=True)
	model = TQC(policy='MlpPolicy', env=env(), learning_starts=1000000, gradient_steps=1, batch_size=512)
	policy_loss = initialize_bc_policy(model, expert_replay_buffer=load_from_pkl('/Users/Yakumo/Downloads/replayBuffer_cubeTorque.pkl', model.verbose), dynamic_model=None, n_iter=100000, batch_size=512)
	print("policy loss", policy_loss)
	model.save('/Users/Yakumo/Downloads/TQC_bc.pkl')

	
def plot(log_dir): # need Monitor
	plot_results([log_dir], num_timesteps=None, x_axis=results_plotter.X_TIMESTEPS, task_name="TD3 LunarLander")
	plt.show()
	
	
if __name__ == '__main__':
	folder = Path(__file__).resolve().parent
	log_path = folder/"log"
	data_path = folder/"data"
	log_path.mkdir(exist_ok=True)
	
	#model_class = TQC; kwargs = None
	#model_class = TQC_RCE; kwargs = dict(demonstration_replay_buffer=data_path/'replay_buffer_reward_on_cube_kuka_success_only.pkl', action_strategy='current policy')
	#model_class = TQC_SQIL; kwargs = dict(demonstration_replay_buffer=data_path/'replay_buffer_reward_on_cube_kuka_single_config.pkl')
	#model_class = TQC_RED; kwargs = dict(demonstration_replay_buffer=data_path/'replay_buffer_reward_on_cube_kuka_single_config.pkl')
	#model_class = TQC_PWIL; kwargs = dict(demonstration_replay_buffer=data_path/'replay_buffer_reward_on_cube_kuka_single_config_pwil.pkl', T=1500)
	
	load_replay_buffer = None#data_path/"replay_buffer_reward_on_cube_kuka_single_config.pkl"
	load = None
	env_kwargs = dict(id='kuka_grasping-v0', display=False, obj='cube', steps_to_roll=1, mode='joint torques')#, npmp_decoder='/Users/Yakumo/Downloads/npmp_decoder')
	
	learn(
		model_class=model_class,
		kwargs=kwargs,
		load=load,
		log_path=log_path,
		device='cpu',
		load_replay_buffer=load_replay_buffer,
		env_kwargs=env_kwargs,
		enjoy=False
	)
	
	#learnReach(log_path=log_path, vec_env=False, mode='joint torques', her=True)

	print('end')
	
