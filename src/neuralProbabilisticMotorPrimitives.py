
import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.distributions import DiagGaussianDistribution, kl_divergence
from multiprocessing import Pool
from functools import partial
from torch.nn.functional import logsigmoid, mse_loss
import gym
import gym_grabbing
from gym_grabbing.envs.utils import MLP, InterpolateKeyPointsGrip, InterpolateKeyPointsGrip
from pathlib import Path
import yaml



ENV_ID = "kuka_grasping-v0"
ENVS = [gym.make(ENV_ID, display=False, obj='cube' ,steps_to_roll=1, mode='joint torques', table_height=None), gym.make(ENV_ID, display=False, obj='cube' ,steps_to_roll=1, mode='joint torques')]



def simulate(ind, demo=False, controller_info=None, add_stop=100, ):
	global ENVS
	env = ENVS[demo]
	env.reset()
	observation = env.info['robot state']

	
	controller = InterpolateKeyPointsGrip(ind, **controller_info, initial=env.get_joint_state())
	states = np.zeros((controller_info['n_iter']+add_stop, env.robot_space.shape[0]))
	actions = np.zeros((controller_info['n_iter']+add_stop, env.action_space.shape[0]))
	l, u, = env.lowerLimits, env.upperLimits
	for k in range(controller_info['n_iter']+add_stop):
		if k == controller_info['n_iter']:
			action_current_pos = np.hstack((env.get_joint_state(position=True, normalized=True), -1))
		action = controller.get_action(k, observation) if k < controller_info['n_iter'] else action_current_pos
		action[:-1] = env.pd_controller.computePD(
			bodyUniqueId=env.robot_id,
			jointIndices=env.joint_ids,
			desiredPositions=l+(np.hstack((action[:-1], env.get_fingers(action[-1])))+1)/2*(u-l),
			desiredVelocities=np.zeros(env.n_joints),
			kps=env.kps,
			kds=env.kds,
			maxForces=env.maxForce,
			timeStep=env.time_step
		)[:-env.n_control_gripper] / env.maxForce[:-env.n_control_gripper]
		action_w_noise = np.array(action)
		action_w_noise[:-1] += np.random.normal(0,0.05 if demo else 0.1, size=(env.n_actions-1,))
		states[k] = observation
		actions[k] = action
		_, reward, done, info = env.step(action_w_noise)
		observation = info['robot state']


	return states, actions

def train_controller(K=5, latent_dim=4, half_batch_size=128, folder=None, epoch=10):
	global ENVS
	env = ENVS[0]
	
	controller_info = {'n_iter':1500, 'nb_keypoints':3, 'genes_per_keypoint': env.n_actions-1}
	individuals = []
	if folder is not None:
		folder_path = Path(folder)
		runs = folder_path.glob("**/run_details.yaml")
		
		for i, run in enumerate(runs):
			with open(run, "r") as f:
				d = yaml.safe_load(f)
			if ENV_ID != d['env id'] or not d['successful']: continue
			individuals.append(np.load(run.parent/"individuals.npz")["genotypes"])
		individuals = np.vstack(individuals)
		if len(individuals) == 0:
			print("no individual found")
		controller_info = d['controller info']
	
	batch_size = half_batch_size*2
	encoder = MLP(
		observation_space=gym.spaces.Box(low=np.repeat(env.robot_space.low,K), high=np.repeat(env.robot_space.high,K), shape=(env.robot_space.shape[0]*K,)),
		action_space=gym.spaces.Box(low=np.inf, high=np.inf, shape=(latent_dim,)),
		net_arch=[64,64],
		output_transform="DiagGaussianDistribution",
	)
	decoder = MLP(
		observation_space=gym.spaces.Box(low=np.inf, high=np.inf, shape=(env.robot_space.shape[0]+latent_dim,)),
		action_space=env.action_space,
		net_arch=[32,32],
		output_transform='Tanh',
	)
	rng = np.random.default_rng()
	#distribution, previous_distribution = DiagGaussianDistribution(latent_dim), DiagGaussianDistribution(latent_dim)
	arange_batch_size = np.arange(batch_size)
	for _ in range(epoch):
		inds = rng.random((batch_size, env.action_space.shape[0]*3-2))*2-1
		if len(individuals)>0:
			inds = np.vstack((rng.choice(individuals, half_batch_size, replace=False), inds[:half_batch_size]))
		states = []#np.zeros((batch_size, max_episode_steps+add_stop, env.robot_space.shape[0]))
		actions = []#np.zeros((batch_size, max_episode_steps+add_stop, env.action_space.shape[0]))
		with Pool() as p:
			for i, (state, action) in enumerate(p.starmap(partial(simulate, controller_info=controller_info, add_stop=100), zip(inds, [i<batch_size for i in range(batch_size)]))):
				states.append(state)#states[i] = state
				actions.append(action)#actions[i] = action
		states = np.array(states)
		actions = np.array(actions)
		random_indexes = [rng.permutation(np.arange(1,states.shape[1]-K, dtype=int)) for _ in range(batch_size)]
		for index in np.array(random_indexes).T: # max_episode_steps-K-1 times
			state_window = states[arange_batch_size[:,None], index[:,None] + np.arange(K+1)]
			state_window = state_window.reshape(batch_size, env.robot_space.shape[0]*(K+1)) # stack futur observations
			state, x = np.split(state_window, [env.robot_space.shape[0]], axis=-1) # x is the futur
			previous_x = state_window[:, :env.robot_space.shape[0]*K]
			action = actions[arange_batch_size,index]
			
			#with th.no_grad(): # yes or no ?
			previous_latent = encoder(th.as_tensor(previous_x, dtype=th.float), deterministic=True)
			previous_distribution = encoder.distribution.distribution
			
			latent = encoder(th.as_tensor(x, dtype=th.float), deterministic=False)
			distribution = encoder.distribution.distribution
			
			prediction = decoder(th.hstack((th.as_tensor(state, dtype=th.float), latent)))
			loss = mse_loss(input=prediction, target=th.as_tensor(action, dtype=th.float)) # reconstruction error
			regularization_kl = th.distributions.kl_divergence(distribution, th.distributions.Normal(0, 0.1)).mean() # regularization
			previous_kl = th.distributions.kl_divergence(distribution, previous_distribution).mean()
			loss += 0.0002*(5*previous_kl+regularization_kl)

			encoder.optimizer.zero_grad()
			decoder.optimizer.zero_grad()
			loss.backward()
			encoder.optimizer.step()
			decoder.optimizer.step()
		print("loss", loss.item(), "regularization kl", regularization_kl.item(), "previous kl", previous_kl.item())
		print("predic", prediction[0].tolist())
		print("action", action[0].tolist())
		print("latent", np.round(latent[0].tolist(), 3).tolist())
		print("previo", np.round(previous_latent[0].tolist(), 3).tolist())
		print("futur", np.round(x[0].tolist(), 3).tolist())
		print("prevf", np.round(previous_x[0].tolist(), 3).tolist())
		encoder.save("/Users/Yakumo/Downloads/npmp_encoder")
		decoder.save("/Users/Yakumo/Downloads/npmp_decoder")
	

if __name__ == '__main__':
	train_controller(folder="/Users/Yakumo/Downloads/kukaPdStable/kukaSingleConfPdStable", epoch=1000)
	print('end')
