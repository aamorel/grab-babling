#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path
import json, yaml
import argparse

import gym
import numpy as np
import gym_grabbing
import imageio
import matplotlib.pyplot as plt
from gym_grabbing.envs.utils import InterpolateKeyPointsGrip

def main(
	path,
	print_close: bool = False,
	plot_trajectories: int = None,
	speed: float = None,
	select: str = "diversity samples",
	object_frame: bool = False,
) -> None:

	folder = Path(path)
	if len(list(folder.glob("**/run_details.yaml"))) == 0:
		print("No run found"); return

	# We suppose the environment is the same for all runs !
	with open(next(folder.glob("**/run_details.yaml")), "r") as f:
		d = yaml.load(f)
	d['env kwargs']['display'] = True
	#d['env kwargs']['fixed_arm'] = False
	env = gym.make(**d['env kwargs'])

	for i in range(env.p.getNumJoints(env.robot_id)): print("joint info", env.p.getJointInfo(env.robot_id, i))
	controller_info = d['controller info']

	# selected individuals to select
	if select == "diversity samples":
		individuals = [np.load(indPath) for indPath in folder.glob("**/type*_0.npy")]
	elif select in {"all", "genotype samples"}:
		selection = {'all': 'genotypes', "genotype samples": "genotype samples"}[select]
		individuals = ([np.load(repertoire)[selection] for repertoire in folder.glob("**/individuals.npz")])
	else:
		print("select must be in {'all', 'diversity samples', 'genotype samples'}.")
		return

	if object_frame:
		lines = [env.p.addUserDebugLine([0, 0, 0], end, color, parentObjectUniqueId=env.obj_id) for end, color in zip(np.eye(3)*0.2, np.eye(3))]

	textid = -1
	timeid = -1
	dt = env.p.getPhysicsEngineParameters()["fixedTimeStep"]
	for i, ind in enumerate(individuals):

		env.p.removeAllUserDebugItems()
		o = env.reset(load='state')#multiply_friction={'lateral':1/1.2, 'rolling':1/10, 'spinning':1/10})
		#env.p.removeBody(env.robot_id); env.p.removeBody(env.obj_id); time.sleep(1000000)
		controller = InterpolateKeyPointsGrip(
			ind,
			**controller_info,
			initial=
				env.get_joint_state(position=True) if d['env kwargs']['mode'] in {'joint positions', 'pd stable'} else
				env.get_joint_state(position=False) if d['env kwargs']['mode'] == 'joint velocities' else None
		)
		maxspeed , maxtorque = 0,0
		textid = env.p.addUserDebugText("", [0.5,0,0], textColorRGB=[1,0,0], replaceItemUniqueId=textid)

		contact_table = False
		opened = True
		last_time = time.perf_counter()
		last_pos = env.info['end effector position']
		for k in range(int(controller_info["n_iter"])): # simulation
			if object_frame:
				lines = [env.p.addUserDebugLine([0, 0, 0], end, color, parentObjectUniqueId=env.obj_id, replaceItemUniqueId=id) for end, color, id in zip(np.eye(3)*0.2, np.eye(3), lines)]

			action = controller.get_action(k, o) #np.random.rand(8)*2-1
			o, r, done, inf = env.step(action)#; print(action)

			# print "close"
			if action[-1]==-1 and opened and print_close:
				textid = env.p.addUserDebugText("close", [0.5,0,0], textColorRGB=[1,0,0], replaceItemUniqueId=textid)
				opened = False

			# plot line
			pos = inf['end effector position']
			if plot_trajectories is not None and k%plot_trajectories==0:
				color = [0, k/controller_info["n_iter"], 1-k/controller_info["n_iter"]]
				env.p.addUserDebugLine(last_pos, pos, color, parentObjectUniqueId=-1, lineWidth=10)
				last_pos = pos

			states = env.p.getJointStates(env.robot_id, env.joint_ids[:-env.n_control_gripper])
			maxspeed = np.max([maxspeed, *[s[1] for s in states]])
			maxtorque  = np.max([maxtorque, np.abs(inf['applied joint motor torques']).max()])
			contact_table = contact_table or len(inf['contact robot table'])>0

			# slow down for viszualization
			if speed is not None:
				now = time.perf_counter()
				time.sleep(max(0,dt*d['env kwargs']['steps_to_roll']-(now-last_time))+speed)
				last_time = now

		print("maxspeed", maxspeed, "maxtorque", maxtorque)
		#time.sleep(1)

	env.reset(load='state')
	#input("press enter to continue")
	env.close()


def record(path):
	ind_path = Path(path)
	ind = np.load(path)
	with open(Path(ind_path).parent/'run_details.yaml', 'r') as f:
		d = yaml.safe_load(f)
	env = gym.make(**d['env kwargs'])

	o = env.reset()

	controller_info = d['controller info']
	controller = InterpolateKeyPointsGrip(ind, **controller_info, initial=env.get_joint_state())
	images = []
	interval = 1
	for k in range(controller_info["n_iter"]):
		o, r, eo, inf = env.step(controller.get_action(k, o))
		if k%interval==0:
			images.append(env.render(mode='rgba_array'))
	imageio.mimsave(ind_path.parent/f"{d['env kwargs']['id'][:-3]}_{d['env kwargs']['obj']}.mp4", images, fps=240/d['env kwargs']['steps_to_roll']/interval)
	env.close()

def recordFirstSignal(path, save_past=100):
	folder = Path(path)
	with open(next(folder.glob("**/run_details.yaml")), 'r') as f:
		d = yaml.safe_load(f)
	env = gym.make(f"{d['robot']}_grasping-v0", display=False, obj=d['object'], steps_to_roll=d['steps to roll'], mode=d['mode'])#, object_position=d['object_position'], object_xyzw=d['object_xyzw'], joint_positions=d['joint_positions'])

	for j, ind_path in enumerate(folder.glob("**/type*.npy")):
		print(ind_path)
		ind = np.load(ind_path)
		env.reset()
		#env.p.removeBody(env.robot_id); env.p.removeBody(env.obj_id); imageio.imwrite(folder/'background.png', env.render(mode='rgba_array')) ; return

		controller_info = d['controller info']
		controller = InterpolateKeyPointsGrip(ind, **controller_info, initial=env.get_joint_state())
		i = 0
		for k in range(controller_info["n_iter"]):
			action = controller.get_action(k)
			o, r, eo, inf = env.step(action)
			if r: # save the first reward signal
				i=k-save_past
				break
		else: # skip this individual
			continue
		# re-simulate to render at the right time
		env.reset()
		controller = InterpolateKeyPointsGrip(
			ind,
			**controller_info,
			initial=
				env.get_joint_state(position=True) if d['mode'] in {'joint positions', 'pd stable'} else
				env.get_joint_state(position=False) if d['mode'] == 'joint velocities' else None
		)
		for k in range(i):
			action = controller.get_action(k)
			o, r, eo, inf = env.step(action)

		imageio.imwrite(folder/f"{d['robot']}_{j}.png", env.render('rgba_array'))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-r", "--runs", help="The directory containing runs", type=str, default=str(Path(__file__).parent.parent/'runs'))
	parser.add_argument("-i", "--individual-selection", help="The individual selection", type=str, default="diversity", choices=["all", "diversity", "genotype"])
	parser.add_argument("-c", "--close", help="Print close", action="store_true")
	parser.add_argument("-o", "--object-frame", help="Enable object frame", action="store_true")
	parser.add_argument("-s", "--speed", help="Enable object frame", type=float, default=None)
	parser.add_argument("-t", "--trajectory", help="Enable object frame", type=int, default=None)
	args = parser.parse_args()
	main(
		path=args.runs,
		print_close=args.close,
		plot_trajectories=args.trajectory,
		speed=args.speed,
		select={"all":"all", "diversity": "diversity samples", "genotype": "genotype samples"}[args.individual_selection],
		object_frame=args.object_frame,
	)
