
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tab indentation
# simulate individuals and store Baxter's joint position in .npy files

import argparse
import glob
from pathlib import Path
import yaml
import time

import gym
import numpy as np
import pybullet as p
import gym_grabbing
from scipy import interpolate

# bad!
#import sys, os
#sys.path.append(str(Path(__file__).parent.parent/"grab-babling"/"src"))
#from controllers import InterpolateKeyPointsEndPauseGripAssumption, InterpolateKeyPointsGrip

class InterpolateKeyPointsGrip():

    def __init__(self, individual, n_iter, genes_per_keypoint, nb_keypoints, initial=None):
        """Interpolate actions between keypoints
        Only one parameter (last gene) for the gripper, specifying the time at which it should close
        """
        assert len(individual) == nb_keypoints * genes_per_keypoint + 1, f"len(individual)={len(individual)} must be equal to nb_keypoints({nb_keypoints}) * genes_per_keypoint({genes_per_keypoint}) + 1(gripper) = {nb_keypoints * genes_per_keypoint + 1}"
        self.n_iter = n_iter
        actions = np.split(np.array(individual[:-1]), nb_keypoints)

        interval_size = int(n_iter / nb_keypoints)
        interp_x = [int(interval_size / 2 + i * interval_size) for i in range(nb_keypoints)]
        if initial is not None: # add initial joint states to get a smooth motion
            assert len(initial) == genes_per_keypoint, f"The length of initial={len(initial)} must be genes_per_keypoint={genes_per_keypoint}"
            actions.insert(0, initial) # the gripper is not included
            interp_x.insert(0, 0)
        self.action_polynome = interpolate.interp1d(interp_x, actions, kind='quadratic', axis=0, bounds_error=False, fill_value='extrapolate')
        self.open_loop = True
        self.grip_time = int((individual[-1]+1)/2 * n_iter)

    def get_action(self, i, _o=None):
        if i <= self.n_iter:
            action = self.action_polynome(i)
            action = np.append(action, 1 if i < self.grip_time else -1)  # gripper: 1=open, -1=closed
            return action
        else: # feed last action
            return self.get_action(self.n_iter)

def main(directory, step=10, supervised=True, n_samples=10):
    """
    Convert genotypes of baxter individuals into joint trajectory

    Parameters
    ----------
    directory: str
    	the directory containing runs, each run contains run_details.json and *npy
    step: positive int
    	at every {step}, joint positions are saved
    supervised: bool
    	if true, display is enable and ask for every individual if it is valid
    	if false, trajectories of all individuals will be saved without displaying
    out: str
    	the directory where trajectory files .npy will be saved
    	if not specified, it will create the folder 'trajectory in the current directory'
    """
    directoryStore = Path(directory)
    #directoryStore.mkdir(exist_ok=True)

    folders = [path.parent for path in Path(directory).glob("**/run_details.yaml")]


    success, eval = 0, 0
    env = None
    textid = -1
    envs = {}
    for folder in folders:
        with open(folder/"run_details.yaml", "r") as f:
            d = yaml.safe_load(f)
        if not d['successful']: continue

        controller_info = d['controller info']
        quality = d['multi quality'] is not None and "+grasp robustness" in {q for qs in d['multi quality'] for q in qs}
        key = f"{d['env kwargs']['obj']}{'Qual' if quality else 'NoQual'}"
        if d['env kwargs']['obj'] not in envs:
            envs[d['env kwargs']['obj']] = gym.make(**d['env kwargs'])#"baxter_grasping-v0", display=supervised, obj=d['env kwargs']['object'], steps_to_roll=10)
        repertoire = np.load(folder/"individuals.npz")['genotypes']
        count_success = 0
        success_ind = []

        for i, ind in enumerate(repertoire):
        #folder.glob("*.npy")): # simulate each individual
            #ind = np.load(indPath)
            trajectory = np.zeros((np.ceil(controller_info['n_iter']/step).astype(int), 7))
            again = True

            while again: # repeat if asked

                env = envs[d['env kwargs']['obj']]
                o = env.reset()

                textid = p.addUserDebugText("", [0.5,0,0], textColorRGB=[1,0,0], replaceItemUniqueId=textid)
                opened = True
                contactTable = False
                controller = InterpolateKeyPointsGrip(ind, **controller_info, initial=env.get_joint_state())
                for j in range(controller_info['n_iter']): # simulation
                    o, r, eo, info = env.step(controller.get_action(j,o))

                    if j%step==0:
                        trajectory[j//step] = info['joint positions'][:-env.n_control_gripper]
                    if opened and j>controller.grip_time:
                        closeTime = j/step
                        opened = False
                        textid = p.addUserDebugText("close", [0.5,0,0], textColorRGB=[1,0,0], replaceItemUniqueId=textid)
                    contactTable = contactTable or len(info['contact robot table'])>0

                assert (trajectory[-1]!=0).all()
                if supervised:
                    while True:
                        x = input("v: validate, r: refuse, a: again ").strip()
                        if x == "v":
                            (directoryStore/f"{key}").mkdir(exist_ok=True)
                            with open(directoryStore/f"{key}"/f"{folder}_{i}", 'wb') as f:
                                np.save(f, trajectory)
                                np.save(f, closeTime)
                            again = False
                        elif x == "r":
                            again = False
                        elif x == "a":
                            again = True
                        if x in {'v','r','a'}:
                            break
                else:
                    if info["is_success"]:#True: # save
                        count_success += 1
                        (directoryStore/f"{key}").mkdir(exist_ok=True)
                        with open(directoryStore/f"{key}/{folder.stem}_{i}", 'wb') as f:
                            np.save(f, trajectory)
                            np.save(f, closeTime)
                        success_ind.append(i)
                    again = False
            eval += 1
            success += r
            if count_success >= n_samples:
                f = folder/"individuals.npz"
                data = dict(np.load(f))
                data['transfer samples'] = np.array(success_ind, dtype=int) # save indexes
                np.savez_compressed(f, **data)
                break # go to the next folder
    print("end", "success", success, "eval", eval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--runs", help="The directory containing runs", type=str, default=str(Path(__file__).parent.parent/'runs'))
    args = parser.parse_args()
    main(directory=args.runs, supervised=False)
