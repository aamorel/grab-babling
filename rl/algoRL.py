
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn.functional import logsigmoid, mse_loss
from scipy.spatial import KDTree

from stable_baselines3.common.save_util import save_to_pkl, load_from_pkl
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import polyak_update, get_device
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit, ReplayBufferSamples
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, ConvertCallback
from stable_baselines3.common import logger
from stable_baselines3 import SAC
from sb3_contrib import TQC
from sb3_contrib.common.utils import quantile_huber_loss

#import pdb

class InverseModel(BaseModel):
	def __init__(self, net_arch, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.net_arch = net_arch
		
		self.features_extractor = self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
		self.model = th.nn.Sequential(*create_mlp(input_dim=self.features_extractor.features_dim*2, output_dim=self.action_space.shape[0], net_arch=net_arch))
		self.optimizer = self.optimizer_class(self.parameters(), lr=1e-3, **self.optimizer_kwargs)
		
	def forward(self, observation: th.Tensor, next_observation: th.Tensor):
		return self.model(th.hstack((self.extract_features(observation), self.extract_features(next_observation))))
		
	def _get_constructor_parameters(self):
		data = super()._get_constructor_parameters()
		data.update({'net_arch':self.net_arch})
		return data
		
class ForwardModel(BaseModel):
	def __init__(self, net_arch, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.net_arch = net_arch
		
		self.features_extractor = self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
		self.model = th.nn.Sequential(*create_mlp(input_dim=self.features_extractor.features_dim+self.action_space.shape[0], output_dim=self.features_extractor.features_dim, net_arch=net_arch))
		self.optimizer = self.optimizer_class(self.parameters(), lr=1e-3, **self.optimizer_kwargs)
		
	def forward(self, observation: th.Tensor, action: th.Tensor):
		return self.model(th.hstack((self.extract_features(observation), self.extract_features(action))))
	
	def _get_constructor_parameters(self):
		data = super()._get_constructor_parameters()
		data.update({'net_arch':self.net_arch})
		return data
		
def train_dynamic_model(replay_buffer, dynamic_model: Union[ForwardModel, InverseModel], n_iter=100000, batch_size=256, env=None):
	for i in range(n_iter):
		replay_data = replay_buffer.sample(batch_size, env=env)
		if isinstance(dynamic_model, InverseModel):
			prediction = dynamic_model(replay_data.observations, replay_data.next_observations)
			target = replay_data.actions
		else: # forward model
			prediction = dynamic_model(replay_data.observations, replay_data.actions)
			target = replay_data.next_observations
		
		dynamic_model_loss = mse_loss(input=prediction, target=target) * (1-replay_data.dones)
		dynamic_model.optimizer.zero_grad()
		dynamic_model_loss.mean().backward()
		dynamic_model.optimizer.step()
		#print(np.round(dynamic_model_loss.mean().item(),3), end=' ')
		#print(replay_data.observations.tolist())
	return dynamic_model_loss.mean().item()
		
def initialize_bc_policy(algo, demonstration_replay_buffer, dynamic_model: Optional[Union[ForwardModel, InverseModel]] = None, n_iter=100000, batch_size=256, env=None):
	for i in range(n_iter):
		replay_data = demonstration_replay_buffer.sample(batch_size, env=env)
		action, log_prob = algo.actor.action_log_prob(replay_data.observations) 
		if isinstance(dynamic_model, ForwardModel):
			prediction = dynamic_model(replay_data.observations, action) # state prediction
			target = replay_data.next_observations
		else:
			prediction = action
			if isinstance(dynamic_model, InverseModel):
				with th.no_grad():
					target = dynamic_model(replay_data.observations, replay_data.next_observations)
			else: # use action from the replay buffer
				target = replay_data.actions
		policy_loss = mse_loss(input=prediction, target=target) * (1-replay_data.dones)
		algo.actor.optimizer.zero_grad()
		policy_loss.mean().backward()
		algo.actor.optimizer.step()
		#print(policy_loss.mean().item(), zip(prediction[0].tolist(), target[0].tolist()))
	return policy_loss.mean().item()

def behaviouralCloningWithModel(collection_timesteps=100000, repeat=10000, use_inverse=False, model=None, demonstration_replay_buffer='', device='auto', save_path=None): # train dynamic model and bc
	env = lambda display=False: gym.make('kuka_grasping-v0', display=False, obj='cube', steps_to_roll=1, mode='joint torques', reset_random_initial_state=True)#, early_stopping=True)
	venv = VecNormalize(DummyVecEnv([env]), norm_obs=True, norm_reward=False)
	if model is None:
		model = TQC(policy='MlpPolicy', env=venv, learning_starts=1000000, gradient_steps=1, device=device)
		
	if hasattr(model, 'dynamic_model'):
		dynamic_model = model.dynamic_model
	else:
		dynamic_model_class = partial(InverseModel if use_inverse else ForwardModel,  net_arch=[256, 256], observation_space=model.observation_space, action_space=model.action_space, features_extractor_class=model.policy.features_extractor_class, features_extractor_kwargs=model.policy.features_extractor_kwargs, normalize_images=model.policy.normalize_images, optimizer_class=model.policy.optimizer_class, optimizer_kwargs=model.policy.optimizer_kwargs, device=device)
		dynamic_model = dynamic_model_class()
	model.learn(0) # setup
	dummyCallback = ConvertCallback(None)
	dummyCallback.init_callback(model)
	model._total_timesteps = 10000000
	for i in range(repeat):
		model.collect_rollouts(
			env=model.env,
			train_freq=TrainFreq(10000, TrainFrequencyUnit("step")),
			action_noise=model.action_noise,
			callback=dummyCallback,
			learning_starts=model.learning_starts,
			replay_buffer=model.replay_buffer,
		)
		
		dynamic_model_loss = train_dynamic_model(replay_buffer=model.replay_buffer, dynamic_model=dynamic_model, n_iter=collection_timesteps, batch_size=512)
		print(i, 'dynamic model', dynamic_model_loss)
		policy_loss = initialize_bc_policy(model, demonstration_replay_buffer=load_from_pkl(demonstration_replay_buffer, model.verbose), dynamic_model=dynamic_model, n_iter=collection_timesteps, batch_size=512)
		print(i, 'policy', policy_loss)
	if save_path is not None:
		model.save(save_path+'/TQC_bc.pkl')
		dynamic_model.save(save_path+f"/{'inverse' if use_inverse else 'forward'}Model.pkl")
	return model, dynamic_model

	
class TQC_dynamic_model(TQC):
	def __init__(self,
		action_strategy='behaviour policy',
		*args, **kwargs
	):
		action_strategy_clean = action_strategy.strip().lower()
		assert action_strategy_clean in {'current policy', 'behaviour policy', 'inverse model', 'forward model'}, 'examples_actions_strategy must be either current policy, behaviour policy, inverse model, forward model'
		super().__init__(*args, **kwargs)
		self.action_strategy = action_strategy_clean
		self.dynamic_model = None
		if self.action_strategy == 'inverse model':
			self.dynamic_model = InverseModel(net_arch=[512, 256], observation_space=self.observation_space, action_space=self.action_space, features_extractor_class=self.policy.features_extractor_class, features_extractor_kwargs=self.policy.features_extractor_kwargs, normalize_images=self.policy.normalize_images, optimizer_class=self.policy.optimizer_class, optimizer_kwargs=self.policy.optimizer_kwargs).to(self.device)
		elif self.action_strategy == 'forward model':
			raise Exception('not implemented')
		self._n_gradients = 0
	
	def step_dynamic_model(self, replay_data):
		if self.action_strategy == 'inverse model':
			prediction = self.dynamic_model(replay_data.observations, replay_data.next_observations)
			target = replay_data.actions
		elif self.action_strategy == 'inverse model': # forward model
			prediction = self.dynamic_model(replay_data.observations, replay_data.actions)
			target = replay_data.next_observations
		else:
			return None
		dynamic_model_loss = mse_loss(input=prediction, target=target) * (1-replay_data.dones)
		dynamic_model_loss = dynamic_model_loss.mean()
		self.dynamic_model.optimizer.zero_grad()
		dynamic_model_loss.backward()
		self.dynamic_model.optimizer.step()
		return dynamic_model_loss.item()
		
	def pretrain(self, collection_timesteps=10000, repeat=1000, demonstration_replay_buffer=None):
		use_dynamic_model = self.action_strategy in {'inverse model', 'forward model'}
		self.learn(0) # setup
		dummyCallback = ConvertCallback(None)
		dummyCallback.init_callback(self)
		self._total_timesteps = collection_timesteps*repeat
		for i in range(repeat):
			self.collect_rollouts(
				env=self.env,
				train_freq=TrainFreq(collection_timesteps, TrainFrequencyUnit("step")),
				action_noise=self.action_noise,
				callback=dummyCallback,
				learning_starts=self.learning_starts,
				replay_buffer=self.replay_buffer,
			)
			if use_dynamic_model:
				dynamic_self_loss = train_dynamic_self(replay_buffer=self.replay_buffer, dynamic_self=self.dynamic_model, n_iter=collection_timesteps, batch_size=self.batch_size)
				print(i, 'dynamic self', dynamic_self_loss)
			if demonstration_replay_buffer is not None:
				policy_loss = initialize_bc_policy(self, demonstration_replay_buffer=load_from_pkl(demonstration_replay_buffer, self.verbose), dynamic_model=self.dynamic_model, n_iter=collection_timesteps, batch_size=self.batch_size)
				print(i, 'policy', policy_loss)
		return self
		
	def _get_constructor_parameters(self):
		data = super()._get_constructor_parameters()
		data.update({'action_strategy':self.action_strategy})
		return data
		
	def _get_torch_save_params(self):
		state_dicts_names, torch_variable_names = super()._get_torch_save_params()
		if isinstance(self.dynamic_model, (InverseModel, ForwardModel)):
			state_dicts_names += ['dynamic_model']
		return state_dicts_names, torch_variable_names

class TQC_RCE(TQC_dynamic_model): # recursive classification of examples https://arxiv.org/pdf/2103.12656v1.pdf
	def __init__(self, demonstration_replay_buffer=None, n_step=10, *args, **kwargs):

		super().__init__(*args, **kwargs)
		if demonstration_replay_buffer is not None:
			self.demonstration_replay_buffer = load_from_pkl(demonstration_replay_buffer)
			self.demonstration_replay_buffer.device = self.device
		else: self.demonstration_replay_buffer = None
		self.n_step = n_step
		self.critic_loss, self.critic_last, self.critic_examples, self.actor_loss, self.entropy = None, None, None, None, None

		
		
	def train(self, gradient_steps: int, batch_size: int = 64) -> None:
		self.n_target_quantiles = self.critic.quantiles_total - self.top_quantiles_to_drop_per_net * self.critic.n_critics
		# Update optimizers learning rate
		optimizers = [self.actor.optimizer, self.critic.optimizer]
		if self.ent_coef_optimizer is not None:
			optimizers += [self.ent_coef_optimizer]

		# Update learning rate according to lr schedule
		self._update_learning_rate(optimizers)

		ent_coef_losses, ent_coefs = [], []
		actor_losses, critic_losses = [], []
		model_losses = []

		for gradient_step in range(gradient_steps):
			# Sample replay buffer
			#replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
			if self.replay_buffer.full:
				batch_inds = (np.random.randint(self.n_step, self.buffer_size, size=batch_size) + self.replay_buffer.pos) % self.buffer_size
			else:
				batch_inds = np.random.randint(0, max(self.replay_buffer.pos-self.n_step,1), size=batch_size)
			obs = self.replay_buffer._normalize_obs(self.replay_buffer.observations[batch_inds, 0, :], self._vec_normalize_env)
			rewards = self.replay_buffer._normalize_reward(self.replay_buffer.rewards[batch_inds], self._vec_normalize_env)
			safe_batch_inds = (batch_inds + 1) % self.buffer_size # batch indexe
			next_obs = self.replay_buffer._normalize_obs(self.replay_buffer.observations[safe_batch_inds, 0, :], self._vec_normalize_env) # obs t+1
			
			n_step, ok = np.ones(batch_size, dtype=int), np.ones(batch_size, dtype=bool) # fetch the greatest n_step value without reaching done
			for i in range(1, self.n_step):
				ok = np.logical_and(np.logical_not(self.replay_buffer.dones[(safe_batch_inds + i) % self.buffer_size, 0]), ok)
				n_step += ok
				
			safe_batch_inds = (batch_inds + n_step) % self.buffer_size
			futur_obs = self.replay_buffer._normalize_obs(self.replay_buffer.observations[safe_batch_inds, 0, :], self._vec_normalize_env) # obs t+n_step
			futur_act = self.replay_buffer.actions[safe_batch_inds, 0, :] # action t+n_step
			# convert to torch
			data = (
				obs,
				self.replay_buffer.actions[batch_inds, 0, :],
				next_obs,
				self.replay_buffer.dones[batch_inds],
				rewards,
			)
			replay_data = ReplayBufferSamples(*tuple(map(self.replay_buffer.to_torch, data)))
			#observations = th.tensor(obs).to(self.device)
			#actions = th.tensor(self.replay_buffer.actions[batch_inds, 0, :]).to(self.device)
			#next_observations = th.tensor(next_obs).to(self.device)
			#dones = th.tensor(self.replay_buffer.dones[batch_inds]).to(self.device)[:,None,None] # * (1 - self.replay_buffer.timeouts[batch_inds])
			dones = th.vstack((replay_data.dones, replay_data.dones)) # repeat for futur
			#rewards = th.tensor(rewards).to(self.device)#[:,None,None]
			futur_observations = th.tensor(futur_obs).to(self.device)
			#futur_actions = th.tensor(futur_act).to(self.device)
			n_step = th.tensor(n_step).to(self.device)#[:,None,None] # batch_size x n_critics x n_quantiles
			# end sample replay buffer
			
			# train inverse model
			#inverse_model_loss = mse_loss(model(observations, next_observations), actions) * (1-dones)
			#self.model.optimizer.zero_grad()
			#inverse_model_loss.backward()
			#self.model.optimizer.step()
			

			# We need to sample because `log_std` may have changed between two gradient steps
			if self.use_sde:
				self.actor.reset_noise()

			# Action by the current actor for the sampled state
			actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
			log_prob = log_prob.reshape(-1, 1)

			ent_coef_loss = None
			if self.ent_coef_optimizer is not None:
				# Important: detach the variable from the graph
				# so we don't change it with other losses
				# see https://github.com/rail-berkeley/softlearning/issues/60
				ent_coef = th.exp(self.log_ent_coef.detach())
				ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
				ent_coef_losses.append(ent_coef_loss.item())
			else:
				ent_coef = self.ent_coef_tensor

			ent_coefs.append(ent_coef.item())
			self.replay_buffer.ent_coef = ent_coef.item()

			# Optimize entropy coefficient, also called
			# entropy temperature or alpha in the paper
			if ent_coef_loss is not None:
				self.ent_coef_optimizer.zero_grad()
				ent_coef_loss.backward()
				self.ent_coef_optimizer.step()

			# RCE ADAPTATION ###########################################################
			#sample_id = th.randperm(len(self.examples))[:batch_size]
			#examples = self.replay_buffer._normalize_obs(self.examples[sample_id], self._vec_normalize_env) # sample examples without replacement
			example_replay_data = self.demonstration_replay_buffer.sample(batch_size=batch_size, env=self._vec_normalize_env)
			
			
			with th.no_grad():
				if self.action_strategy == 'inverse model':
					
					example_next_states = self.replay_buffer._normalize_obs(self.example_next_states[sample_id], self._vec_normalize_env)
					example_actions = self.dynamic_model(examples, example_next_states)
				elif self.action_strategy == 'current policy':
					example_actions, example_log_prob = self.actor.action_log_prob(example_replay_data.observations) # get the action given the example
					
				else:
					example_actions = example_replay_data.actions#self.example_actions[sample_id]
				# Select action according to policy
				next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
				futur_actions, futur_log_prob = self.actor.action_log_prob(futur_observations)

				next_futur_observations = th.vstack((replay_data.next_observations, futur_observations)) # stack next and futur for efficency
				next_futur_actions = th.vstack((next_actions, futur_actions))
				next_futur_quantiles = self.critic_target(next_futur_observations, next_futur_actions)
				next_futur_quantiles = th.sigmoid(next_futur_quantiles) # convert to probability

				# Sort and drop top k quantiles to control overestimation.
				#next_futur_quantiles, _ = th.sort(next_futur_quantiles.reshape(2, batch_size, -1))
				#next_futur_quantiles = next_futur_quantiles[:, :, :self.n_target_quantiles]
				#next_futur_w = (1 - dones)*next_futur_quantiles[:,:,None,None,:] + dones*rewards
				#next_futur_w = next_futur_w / (1.000001-next_futur_w)
				#next_futur_γw = self.gamma**th.vstack((th.ones_like(n_step), n_step)).view(2,batch_size,1,1,1)*next_futur_w
				#y = (next_futur_γw/(next_futur_γw+1)).mean(0)
				#w, futur_w = next_futur_w
				
				next_futur_quantiles, _ = th.sort(next_futur_quantiles.reshape(2*batch_size, -1))
				next_futur_quantiles = next_futur_quantiles[:, :self.n_target_quantiles]
				next_futur_w = ((1 - dones)*next_futur_quantiles)[:,None,None,:]# + dones*rewards
				next_futur_w = next_futur_w / (1.000001-next_futur_w)
				next_futur_γw = self.gamma**th.vstack((th.ones_like(n_step), n_step)).view(2*batch_size,1,1,1)*next_futur_w
				next_γw, futur_γw = th.tensor_split(next_futur_γw, 2)
				y = (next_γw/(next_γw+1) + futur_γw/(futur_γw+1)) / 2
				w, futur_w = th.tensor_split(next_futur_w, 2)


				
			# Get current Quantile estimates using action from the replay buffer
			current_quantiles = self.critic(replay_data.observations, replay_data.actions).unsqueeze(-1) # make it brodcastable to do pairwise operations (double sum)
			example_quantiles = self.critic(example_replay_data.observations, example_actions) # don't need to broadcast
			# Compute critic loss, not summing over the quantile dimension as in the paper.
			example_loss = -(1-self.gamma) * logsigmoid(example_quantiles) #+ max(0,1-self.num_timesteps/1000000)*logsigmoid(-current_quantiles).mean())
			experience_loss = -(1+self.gamma*w) * (y*logsigmoid(current_quantiles) + (1-y)*logsigmoid(-current_quantiles))
			#quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False)
			critic_loss = example_loss.mean() + experience_loss.mean()
			critic_losses.append(critic_loss.item())

			# Optimize the critic
			self.critic.optimizer.zero_grad()
			critic_loss.backward()
			self.critic.optimizer.step()
			
			if self.action_strategy in {'inverse model', 'forward model'}:
				model_losses.append(self.step_dynamic_model(replay_data))
			
			# END RCE ADAPTATION #########################################################################

			# Compute actor loss
			qf_pi = self.critic(replay_data.observations, actions_pi).mean(dim=2).mean(dim=1, keepdim=True)
			actor_loss = (ent_coef * log_prob - qf_pi).mean()
			actor_losses.append(actor_loss.item())

			# Optimize the actor
			self.actor.optimizer.zero_grad()
			actor_loss.backward()
			self.actor.optimizer.step()

			# Update target networks
			self._n_gradients += 1
			if self._n_gradients % self.target_update_interval == 0:
				polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

		self._n_updates += gradient_steps

		self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
		self.logger.record("train/ent_coef", np.mean(ent_coefs))
		self.logger.record("train/actor_loss", np.mean(actor_losses))
		self.logger.record("train/critic_loss", np.mean(critic_losses))
		if len(ent_coef_losses) > 0:
			self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
		if len(model_losses) > 0:
			self.logger.record("train/model_loss", np.mean(model_losses))

	def _excluded_save_params(self) -> List[str]:
		return super()._excluded_save_params() + ["demonstration_replay_buffer"]

class TQC_SQIL(TQC_dynamic_model): # https://arxiv.org/pdf/1905.11108.pdf
	def __init__(self, demonstration_replay_buffer=None, *args, **kwargs):

		super().__init__(*args, **kwargs)
		if demonstration_replay_buffer is None:
			self.demonstration_replay_buffer = None
		else:
			self.demonstration_replay_buffer = load_from_pkl(demonstration_replay_buffer, self.verbose)
			self.demonstration_replay_buffer.device = self.device

		self.n_target_quantiles = self.critic.quantiles_total - self.top_quantiles_to_drop_per_net * self.critic.n_critics

		
	def train(self, gradient_steps: int, batch_size: int = 64) -> None:
		# Update optimizers learning rate
		optimizers = [self.actor.optimizer, self.critic.optimizer]
		if self.ent_coef_optimizer is not None:
			optimizers += [self.ent_coef_optimizer]

		# Update learning rate according to lr schedule
		self._update_learning_rate(optimizers)

		ent_coef_losses, ent_coefs = [], []
		actor_losses, critic_losses = [], []
		model_losses = []

		for gradient_step in range(gradient_steps):
			# Sample replay buffer
			replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
			expert_replay_data = self.demonstration_replay_buffer.sample(batch_size, env=self._vec_normalize_env)


			# We need to sample because `log_std` may have changed between two gradient steps
			if self.use_sde:
				self.actor.reset_noise()

			# Action by the current actor for the sampled state
			actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
			log_prob = log_prob.reshape(-1, 1)

			ent_coef_loss = None
			if self.ent_coef_optimizer is not None:
				# Important: detach the variable from the graph
				# so we don't change it with other losses
				# see https://github.com/rail-berkeley/softlearning/issues/60
				ent_coef = th.exp(self.log_ent_coef.detach())
				ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
				ent_coef_losses.append(ent_coef_loss.item())
			else:
				ent_coef = self.ent_coef_tensor

			ent_coefs.append(ent_coef.item())
			self.replay_buffer.ent_coef = ent_coef.item()

			# Optimize entropy coefficient, also called
			# entropy temperature or alpha in the paper
			if ent_coef_loss is not None:
				self.ent_coef_optimizer.zero_grad()
				ent_coef_loss.backward()
				self.ent_coef_optimizer.step()

			# SQIL ADAPTATION ###########################################################

			with th.no_grad():
				# Select action according to policy
				next_observations = th.vstack((replay_data.next_observations, expert_replay_data.next_observations)) # stack next_observations
				next_actions, next_log_prob = self.actor.action_log_prob(next_observations)
				# Compute and cut quantiles at the next state
				# batch x nets x quantiles
				next_quantiles = self.critic_target(next_observations, next_actions)

				# Sort and drop top k quantiles to control overestimation.
				n_target_quantiles = self.critic.quantiles_total - self.top_quantiles_to_drop_per_net * self.critic.n_critics
				next_quantiles, _ = th.sort(next_quantiles.reshape(2*batch_size, -1)) # 2 times the batch size
				next_quantiles = next_quantiles[:, :n_target_quantiles]

				# td error + entropy term
				target_quantiles = next_quantiles - ent_coef * next_log_prob.reshape(-1, 1)
				rewards = th.vstack((replay_data.rewards, expert_replay_data.rewards)) # stack rewards
				dones = th.vstack((replay_data.dones, expert_replay_data.dones)) # stack dones
				target_quantiles = rewards + (1 - dones) * self.gamma * target_quantiles
				# Make target_quantiles broadcastable to (batch_size, n_critics, n_target_quantiles).
				target_quantiles.unsqueeze_(dim=1)

				# Get current Quantile estimates using action from the replay buffer
				if self.action_strategy == 'inverse model': # use inverse model
					expert_actions = self.dynamic_model(expert_replay_data.observations, expert_replay_data.next_observations)
				elif self.action_strategy == 'behaviour policy': # use action from the replay buffer, the standard way
					expert_actions = expert_replay_data.actions
				elif self.action_strategy == 'current policy': # use current policy
					expert_actions = self.actor.action_log_prob(expert_replay_data.observations)[0]
				actions = th.vstack((replay_data.actions, expert_actions)) # stack actions
				observations = th.vstack((replay_data.observations, expert_replay_data.observations)) # stack observations
			current_quantiles = self.critic(observations, actions)
			# Compute critic loss, not summing over the quantile dimension as in the paper.
			critic_loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False) # self.compute_loss(replay_data, action_strategy='behaviour policy') + self.compute_loss(expert_replay_data, self.action_strategy)
			#print(rewards.mean().item(), next_quantiles.mean().item(), next_log_prob.reshape(-1, 1).mean().item(), target_quantiles.mean().item(), current_quantiles.mean().item(), critic_loss.mean().item())
			#print("rewards", rewards.squeeze(), rewards.size())
			#print("target quantile", target_quantiles.size())
			
			#critic_loss = self.compute_loss(replay_data, action_strategy='behaviour policy', ent_coef=ent_coef) + self.compute_loss(expert_replay_data, self.action_strategy, ent_coef=ent_coef)

			critic_losses.append(critic_loss.item())
			
			if self.action_strategy in {'inverse model', 'forward model'}:
				model_losses.append(self.step_dynamic_model(replay_data))
			
			
			
			# END SQIL ADAPTATION #########################################################################

			# Optimize the critic
			self.critic.optimizer.zero_grad()
			critic_loss.backward()
			self.critic.optimizer.step()
			

			# Compute actor loss
			qf_pi = self.critic(replay_data.observations, actions_pi).mean(dim=2).mean(dim=1, keepdim=True)
			actor_loss = (ent_coef * log_prob - qf_pi).mean()
			actor_losses.append(actor_loss.item())

			# Optimize the actor
			self.actor.optimizer.zero_grad()
			actor_loss.backward()
			self.actor.optimizer.step()

			# Update target networks
			self._n_gradients += 1
			if self._n_gradients % self.target_update_interval == 0:
				polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

		self._n_updates += gradient_steps

		self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
		self.logger.record("train/ent_coef", np.mean(ent_coefs))
		self.logger.record("train/actor_loss", np.mean(actor_losses))
		self.logger.record("train/critic_loss", np.mean(critic_losses))
		if len(ent_coef_losses) > 0:
			self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
		if len(model_losses) > 0:
			self.logger.record("train/model_loss", np.mean(model_losses))
	

		
	def _excluded_save_params(self) -> List[str]:
		return super()._excluded_save_params() + ["demonstration_replay_buffer"]
		
	def compute_loss(self, replay_data, action_strategy, ent_coef):
		with th.no_grad():
			# Select action according to policy
			next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
			# Compute and cut quantiles at the next state
			# batch x nets x quantiles
			next_quantiles = self.critic_target(replay_data.next_observations, next_actions)

			# Sort and drop top k quantiles to control overestimation.
			n_target_quantiles = self.critic.quantiles_total - self.top_quantiles_to_drop_per_net * self.critic.n_critics
			next_quantiles, _ = th.sort(next_quantiles.reshape(replay_data.observations.size()[0], -1))
			next_quantiles = next_quantiles[:, :n_target_quantiles]

			# td error + entropy term
			target_quantiles = next_quantiles - ent_coef * next_log_prob.reshape(-1, 1)
			target_quantiles = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_quantiles
			# Make target_quantiles broadcastable to (batch_size, n_critics, n_target_quantiles).
			target_quantiles.unsqueeze_(dim=1)
			
			if action_strategy == 'inverse model': # use inverse model
				actions = self.dynamic_model(replay_data.observations, replay_data.next_observations)
			elif action_strategy == 'behaviour policy': # use action from the replay buffer, the standard way
				actions = replay_data.actions
			elif action_strategy == 'current policy': # use current policy
				actions = self.actor.action_log_prob(replay_data.observations)[0]

		# Get current Quantile estimates using action from the replay buffer
		current_quantiles = self.critic(replay_data.observations, actions)
		# Compute critic loss, not summing over the quantile dimension as in the paper.
		return quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False)
		
class RandomNetworkDistillation(BaseModel):
	def __init__(self, use_actions=False, output_dim=64, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.use_actions = use_actions
		self.features_extractor = self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
		input_dim = self.features_extractor.features_dim
		if use_actions:
			input_dim += self.action_space.shape[0]
		self.train_net = self.model = th.nn.Sequential(*create_mlp(input_dim=input_dim, output_dim=output_dim, net_arch=[32,32])).to(self.device)
		self.optimizer = self.optimizer_class(self.parameters(), lr=1e-3, **self.optimizer_kwargs)
		self.rand_net = self.model = th.nn.Sequential(*create_mlp(input_dim=input_dim, output_dim=output_dim, net_arch=[ 64])).to(self.device).requires_grad_(requires_grad=False)
		
	
	def train_step(self, observations, actions, dones=0):
		loss = self.forward(observations, actions) * (1-dones)
		loss = loss.mean()
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.item()
			
	def forward(self, states, actions=None):
		data = states.clone().detach().to(self.device)
		if self.use_actions:
			assert actions.size()[1] == self.action_space.shape[0], "actions shape does not match the action space"
			data = th.hstack((data, actions))
		return mse_loss(input=self.train_net(data), target=self.rand_net(data))
		
		
class TQC_RED (TQC_dynamic_model): # https://arxiv.org/pdf/1905.06750.pdf
	def __init__(self, demonstration_replay_buffer, use_actions=False, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.demonstration_replay_buffer = load_from_pkl(demonstration_replay_buffer)
		self.demonstration_replay_buffer.device = self.device
		self.use_actions = use_actions
		self.rnd = RandomNetworkDistillation(
			use_actions=use_actions,
			observation_space=self.observation_space,
			action_space=self.action_space,
			features_extractor_class=self.policy.features_extractor_class,
			features_extractor_kwargs=self.policy.features_extractor_kwargs,
			normalize_images=self.policy.normalize_images,
			optimizer_class=self.policy.optimizer_class,
			optimizer_kwargs=self.policy.optimizer_kwargs,
			).to(self.device)


	def learn(self, total_timesteps, *args, **kwargs):
		dones = 0
		loss = None
		for i in range(int(total_timesteps/1000.)):
			replay_data = self.demonstration_replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
			if self.action_strategy == 'inverse model':
				actions = self.dynamic_model(replay_data.observations, replay_data.next_observations)
				dones = replay_data.dones
			elif self.action_strategy == 'behaviour policy':
				actions = replay_data.actions
			elif self.action_strategy == 'current policy':
				actions, _ = self.actor.action_log_prob(replay_data.observations)
			loss = self.rnd.train_step(replay_data.observations, actions, dones)
		print("RND loss:", loss)
		
		super().learn(*args, **kwargs, total_timesteps=total_timesteps)
		
	
	def train(self, gradient_steps: int, batch_size: int = 64) -> None:
		# Update optimizers learning rate
		optimizers = [self.actor.optimizer, self.critic.optimizer]
		if self.ent_coef_optimizer is not None:
			optimizers += [self.ent_coef_optimizer]

		# Update learning rate according to lr schedule
		self._update_learning_rate(optimizers)

		ent_coef_losses, ent_coefs = [], []
		actor_losses, critic_losses = [], []
		model_losses = []

		for gradient_step in range(gradient_steps):
			# Sample replay buffer
			replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)


			# We need to sample because `log_std` may have changed between two gradient steps
			if self.use_sde:
				self.actor.reset_noise()

			# Action by the current actor for the sampled state
			actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
			log_prob = log_prob.reshape(-1, 1)
			self.entropy = log_prob.mean().item()

			ent_coef_loss = None
			if self.ent_coef_optimizer is not None:
				# Important: detach the variable from the graph
				# so we don't change it with other losses
				# see https://github.com/rail-berkeley/softlearning/issues/60
				ent_coef = th.exp(self.log_ent_coef.detach())
				ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
				ent_coef_losses.append(ent_coef_loss.item())
			else:
				ent_coef = self.ent_coef_tensor

			ent_coefs.append(ent_coef.item())
			self.replay_buffer.ent_coef = ent_coef.item()

			# Optimize entropy coefficient, also called
			# entropy temperature or alpha in the paper
			if ent_coef_loss is not None:
				self.ent_coef_optimizer.zero_grad()
				ent_coef_loss.backward()
				self.ent_coef_optimizer.step()

			# RED ADAPTATION ###########################################################

			with th.no_grad():
				# Select action according to policy
				next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
				# Compute and cut quantiles at the next state
				# batch x nets x quantiles
				next_quantiles = self.critic_target(replay_data.next_observations, next_actions)

				# Sort and drop top k quantiles to control overestimation.
				n_target_quantiles = self.critic.quantiles_total - self.top_quantiles_to_drop_per_net * self.critic.n_critics
				next_quantiles, _ = th.sort(next_quantiles.reshape(batch_size, -1)) # 2 times the batch size
				next_quantiles = next_quantiles[:, :n_target_quantiles]

				# td error + entropy term
				target_quantiles = next_quantiles - ent_coef * next_log_prob.reshape(-1, 1)
				rewards = -self.rnd(replay_data.observations, replay_data.actions)*100 # intrinsic rewards
				#rewards = th.maximum(rewards, replay_data.rewards) # take the max to mix instrinsic and extrinsic, supposing replay_data.rewards is binary
				target_quantiles = rewards + (1 - replay_data.dones) * self.gamma * target_quantiles
				# Make target_quantiles broadcastable to (batch_size, n_critics, n_target_quantiles).
				target_quantiles.unsqueeze_(dim=1)

			# Get current Quantile estimates using action from the replay buffer
			current_quantiles = self.critic(replay_data.observations, replay_data.actions)
			# Compute critic loss, not summing over the quantile dimension as in the paper.
			critic_loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False)

			critic_losses.append(critic_loss.item())

			if self.action_strategy in {'inverse model', 'forward model'}:
				model_losses.append(self.step_dynamic_model(replay_data))
			if self.action_strategy == 'inverse model' and self.use_actions: # retrain once the rnd
				demo_replay_data = self.demonstration_replay_buffer.sample(batch_size, env=self._vec_normalize_env)
				actions = self.dynamic_model(demo_replay_data.observations, demo_replay_data.next_observations)
				self.rnd.train_step(demo_replay_data.observations, actions, demo_replay_data.dones)
				
			
			# END RED ADAPTATION #########################################################################

			# Optimize the critic
			self.critic.optimizer.zero_grad()
			critic_loss.backward()
			self.critic.optimizer.step()
			

			# Compute actor loss
			qf_pi = self.critic(replay_data.observations, actions_pi).mean(dim=2).mean(dim=1, keepdim=True)
			actor_loss = (ent_coef * log_prob - qf_pi).mean()
			actor_losses.append(actor_loss.item())
			self.actor_loss = actor_loss.item()

			# Optimize the actor
			self.actor.optimizer.zero_grad()
			actor_loss.backward()
			self.actor.optimizer.step()

			# Update target networks
			self._n_gradients += 1
			if self._n_gradients % self.target_update_interval == 0:
				polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

		self._n_updates += gradient_steps

		self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
		self.logger.record("train/ent_coef", np.mean(ent_coefs))
		self.logger.record("train/actor_loss", np.mean(actor_losses))
		self.logger.record("train/critic_loss", np.mean(critic_losses))
		if len(ent_coef_losses) > 0:
			self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
	
	def _excluded_save_params(self) -> List[str]:
		return super()._excluded_save_params() + ["demonstration_replay_buffer"]

"""
	def _store_transition(
		self,
		replay_buffer: ReplayBuffer,
		buffer_action: np.ndarray,
		new_obs: np.ndarray,
		reward: np.ndarray, # the reward must be binary: 0 or 1
		done: np.ndarray,
		infos: List[Dict[str, Any]],
	) -> None:
		
		# might not be a good idea: the given examples can be diluted if the policy use one optimal policy, there could be a lack of diversity
		#  add 2 observations to the example dataset per episode if the task is successfull: the firts reward signal and the last observation (if there is still a reward)
		if reward and done: # add the 2 observations
			self.examples = th.vstack((self.examples, self.first_grasp, th.from_numpy(new_obs).to(self.device)))
		elif reward and self.first_grasp is None: # save the first reward signal
			self.first_grasp = th.from_numpy(new_obs).to(self.device)
		elif done: # forget the first reward signal
			self.first_grasp = None
		super()._store_transition(replay_buffer, buffer_action, new_obs, reward, done, infos)
		
		#print(buffer_action)
		# modified the transitions of the buffer for n_step: reward is action(t+nstep) and done is obs(t+n_step)
		if self.n_step == 1:
			super()._store_transition(replay_buffer, buffer_action, new_obs, reward, done, infos)
		elif done==True: # process all transitions left
			b, a, o, r, d, i = self.n_step_buffer[self.n_step_counter]
			a = np.concatenate((a, buffer_action), axis=0)
			o = np.concatenate((o, new_obs), axis=0)
			super()._store_transition(b, a, o, r, d, i)
			#for j in range(self.n_step-1):
				#b, a, o, r, d, i = self.n_step_buffer[(j+self.n_step_counter) % (self.n_step-1)]
				#super()._store_transition(b, a, o, r, d, i)
			self.n_step_counter = -(self.n_step-1) # re-initialize to fill
		elif self.n_step_counter < 0: # fill the n_step_buffer
			self.n_step_buffer[self.n_step_counter+self.n_step-1] = replay_buffer, buffer_action, new_obs, reward, done, infos
			self.n_step_counter += 1
		else: # roll the n_step_buffer
			b, a, o, r, d, i = self.n_step_buffer[self.n_step_counter]
			a = np.concatenate((a, buffer_action), axis=0)
			o = np.concatenate((o, new_obs), axis=0)
			super()._store_transition(b, a, o, r, d, i) # reward is the n_step action, done is the n_step obeservation
			self.n_step_buffer[self.n_step_counter] = replay_buffer, buffer_action, new_obs, reward, done, infos # replace old
			self.n_step_counter = (self.n_step_counter+1) % (self.n_step-1)
"""

class RandomNetwork(BaseModel):
	def __init__(self, use_actions=True, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.use_actions = use_actions
		self.features_extractor = self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
		input_dim = self.features_extractor.features_dim
		if use_actions:
			input_dim += self.action_space.shape[0]
		self.rand_net = self.model = th.nn.Sequential(*create_mlp(input_dim=input_dim, output_dim=64, net_arch=[32,])).to(self.device).requires_grad_(requires_grad=False)
		

	def forward(self, states, actions=None):
		data = states.clone().detach().to(self.device)
		if self.use_actions:
			assert actions.size()[1] == self.action_space.shape[0], "actions shape does not match the action space"
			data = th.hstack((data, actions))
		return self.rand_net(data)

#import time
class TQC_PWIL(TQC):
	# https://arxiv.org/pdf/2006.04678.pdf
	def __init__(self, demonstration_replay_buffer=None, T=None, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.random_network = RandomNetwork(
			use_actions=True,
			observation_space=self.observation_space,
			action_space=self.action_space,
			features_extractor_class=self.policy.features_extractor_class,
			features_extractor_kwargs=self.policy.features_extractor_kwargs,
			normalize_images=self.policy.normalize_images,
			optimizer_class=self.policy.optimizer_class,
			optimizer_kwargs=self.policy.optimizer_kwargs,
		).to(self.device)
		self.T = T
		self.demonstration_replay_buffer = None
		if demonstration_replay_buffer is not None:
			self.demonstration_replay_buffer = load_from_pkl(demonstration_replay_buffer)
			self.demonstration_replay_buffer.device = self.device
			#print(self.demonstration_replay_buffer.observations.shape,self.demonstration_replay_buffer.actions.shape)
			data = np.concatenate((self.demonstration_replay_buffer.observations,self.demonstration_replay_buffer.actions), axis=-1)
			self.demonstrations = self.random_network.forward(
				th.from_numpy(self.demonstration_replay_buffer.observations.squeeze(1)).to(self.device),
				th.from_numpy(self.demonstration_replay_buffer.actions.squeeze(1)).to(self.device)
			)
			#self.demonstrations = KDTree(self.demonstrations.detach().numpy())
			self.sub_demonstration = self.demonstrations.detach().clone().to(self.device)
			#self.visited = np.array([]).reshape(-1,self.demonstrations.m)
			self.D = len(data)
			self.we = th.ones(self.D).to(self.device) / self.D
		#self.c = 0
		

	def _store_transition(
		self,
		replay_buffer: ReplayBuffer,
		buffer_action: np.ndarray,
		new_obs: np.ndarray,
		reward: np.ndarray,
		done: np.ndarray,
		infos: List[Dict[str, Any]],
	) -> None: # the reward from the environment is replaced with the intrinsic PWIL reward
		#print(self._last_obs)
		embedding = self.random_network.forward(th.from_numpy(self._last_obs).to(self.device), th.from_numpy(buffer_action).to(self.device))#.numpy()
		c = 0
		#start = time.perf_counter()
		# PWIL: compare to the whole support
		wπ = 1/self.T
		dist = th.norm(self.sub_demonstration - embedding, dim=-1)
		argsort = th.argsort(dist)
		i = 0
		while wπ > 0:
			j = argsort[i]
			d, we = dist[j].item(), self.we[j].item()
			
			if wπ >= we and i<int(len(self.demonstrations)/self.T)-1:
				c += we*d
				wπ -= we
			else:
				c += wπ*d
				we -= wπ
				wπ = 0
			i += 1
		to_keep = (th.arange(self.sub_demonstration.size(0))[:,None]!=argsort[th.arange(i)]).all(-1)
		self.sub_demonstration = self.sub_demonstration[to_keep] # pop visited
		self.we = self.we[to_keep]
		
		# simple knn version (won't compare to the whole support)
#		for j in range(3):
#			# find the nearest not in visited
#			distances, indices = self.demonstrations.query(embedding, k=self.T)
#			for d,i in zip(distances, indices):
#				# if the neighbour at index i has not been visited
#				if not (self.demonstrations.data[i] == self.visited).all(1).any():
#					self.visited = np.vstack((self.visited, self.demonstrations.data[i])) # append
#					break
#			else:
#				print("WARNING: all neighbours have been visited")
#			c += d
		
		#reward_ = reward if reward>0 else -c*1000
		reward_ = -c*1000
		#print(reward_, i)

		# Store only the unnormalized version
		if self._vec_normalize_env is not None:
			new_obs_ = self._vec_normalize_env.get_original_obs()
			#reward_ = self._vec_normalize_env.get_original_reward()
		else:
			# Avoid changing the original ones
			#self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward
			self._last_original_obs, new_obs_, = self._last_obs, new_obs,

		# As the VecEnv resets automatically, new_obs is already the
		# first observation of the next episode
		if done and infos[0].get("terminal_observation") is not None:
			next_obs = infos[0]["terminal_observation"]
			# VecNormalize normalizes the terminal observation
			if self._vec_normalize_env is not None:
				next_obs = self._vec_normalize_env.unnormalize_obs(next_obs)
		else:
			next_obs = new_obs_
		
		if done:
			self.sub_demonstration = self.demonstrations.detach().clone().to(self.device)
			self.we = th.ones(self.D).to(self.device) / self.D
			#self.visited = np.array([]).reshape(-1,self.demonstrations.m) # empty it

		replay_buffer.add(
			self._last_original_obs,
			next_obs,
			buffer_action,
			reward_,
			done,
			infos,
		)

		self._last_obs = new_obs
		# Save the unnormalized observation
		if self._vec_normalize_env is not None:
			self._last_original_obs = new_obs_
	
	def _excluded_save_params(self) -> List[str]:
		return super()._excluded_save_params() + ["demonstration_replay_buffer"]


class SAC_RCE(SAC): # recursive classification of examples
	def __init__(self, examples=np.array([]), *args, **kwargs): # how to do n_step with a ReplayBuffer?
		super().__init__(*args, **kwargs)
		self.examples = th.from_numpy(examples).to(self.device)
		self._n_gradients = 0
		
	def train(self, gradient_steps: int, batch_size: int = 64) -> None:
		# Update optimizers learning rate
		optimizers = [self.actor.optimizer, self.critic.optimizer]
		if self.ent_coef_optimizer is not None:
			optimizers += [self.ent_coef_optimizer]

		# Update learning rate according to lr schedule
		self._update_learning_rate(optimizers)

		ent_coef_losses, ent_coefs = [], []
		actor_losses, critic_losses = [], []

		for gradient_step in range(gradient_steps):
			# Sample replay buffer
			replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

			# We need to sample because `log_std` may have changed between two gradient steps
			if self.use_sde:
				self.actor.reset_noise()

			# Action by the current actor for the sampled state
			actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
			log_prob = log_prob.reshape(-1, 1)

			ent_coef_loss = None
			if self.ent_coef_optimizer is not None:
				# Important: detach the variable from the graph
				# so we don't change it with other losses
				# see https://github.com/rail-berkeley/softlearning/issues/60
				ent_coef = th.exp(self.log_ent_coef.detach())
				ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
				ent_coef_losses.append(ent_coef_loss.item())
			else:
				ent_coef = self.ent_coef_tensor

			ent_coefs.append(ent_coef.item())

			# Optimize entropy coefficient, also called
			# entropy temperature or alpha in the paper
			if ent_coef_loss is not None:
				self.ent_coef_optimizer.zero_grad()
				ent_coef_loss.backward()
				self.ent_coef_optimizer.step()

			examples = self.examples[th.randperm(len(self.examples))[:batch_size]] # sample examples without replacement
			with th.no_grad():
				# Select action according to policy
				next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
				example_actions, example_log_prob = self.actor.action_log_prob(examples)
				# Compute the next Q values: min over all critics targets
				next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
				next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
				next_q_values = th.sigmoid(next_q_values)
				# add entropy term
				#next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
				# td error + entropy term
				#target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
				w = next_q_values / (1.000001-next_q_values)
				y = self.gamma*w / (self.gamma*w+1) # how to do n-step? special replay buffer?

			# Get current Q-values estimates for each critic network
			# using action from the replay buffer
			current_q_values = self.critic(replay_data.observations, replay_data.actions)
			example_q_values = self.critic(examples, example_actions) 
			example_loss = -(1-self.gamma) * (logsigmoid(example_q_values[0]) + logsigmoid(example_q_values[1]))/2
			experience_loss = -(1+self.gamma*w) * ((y*logsigmoid(current_q_values[0]) + (1-y)*logsigmoid(-current_q_values[0])) + (y*logsigmoid(current_q_values[1]) + (1-y)*logsigmoid(-current_q_values[1])))/2 # the expression ca be simplefied to eq.8 if using 1-step

			critic_loss = example_loss.mean() + experience_loss.mean()
			critic_losses.append(critic_loss.item())

			# Optimize the critic
			self.critic.optimizer.zero_grad()
			critic_loss.backward()
			self.critic.optimizer.step()

			# Compute actor loss
			# Alternative: actor_loss = th.mean(log_prob - qf1_pi)
			# Mean over all critic networks
			q_values_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)
			min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
			actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
			actor_losses.append(actor_loss.item())

			# Optimize the actor
			self.actor.optimizer.zero_grad()
			actor_loss.backward()
			self.actor.optimizer.step()

			# Update target networks
			self._n_gradients += 1
			if self._n_gradients % self.target_update_interval == 0:
				polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

		self._n_updates += gradient_steps

		self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
		self.logger.record("train/ent_coef", np.mean(ent_coefs))
		self.logger.record("train/actor_loss", np.mean(actor_losses))
		self.logger.record("train/critic_loss", np.mean(critic_losses))
		if len(ent_coef_losses) > 0:
			self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


