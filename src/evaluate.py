#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import glob
import sys, os
from multiprocessing import Pool
from functools import partial
from pathlib import Path
import json, yaml
import argparse
import ast

import numpy as np
import gym
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import interpolate
import gym_grabbing
from scoop import futures

from gym_grabbing.envs.utils import InterpolateKeyPointsGrip

sns.set_theme(style="whitegrid")

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--evaluation", help="The evaluation to run", type=str, default="noise", choices=['noise', 'object'])
parser.add_argument("-e", "--environment", help="The robot to use", type=str, default="baxter", choices=["baxter", "kuka", "pepper"])
parser.add_argument("-n", "--n", help="Number of times to repeat an evaluation", type=int, default=20)
parser.add_argument("-i", "--intervals", help="num=intervals in evaluateRobustness", type=int, default=20)
parser.add_argument("-σ", "--σ", help="The noise", type=float, default=0.03)
parser.add_argument("-s", "--select", help="The strategy to select individuals", type=lambda x: int(x) if x.isdigit() else x, default=20)
parser.add_argument("-r", "--runs", help="The run folder", type=str, default=str(Path(__file__).parent.parent/"runs"))
parser.add_argument("-k", "--keep", help="The dictionary specifying specific runs to keep", type=lambda x: ast.literal_eval(x), default=None)
args = parser.parse_args()

if args.environment == "baxter":
    ENVS = {obj:gym.make('baxter_grasping-v0', display=False, obj=obj, steps_to_roll=10, fixed_arm=True) for obj in ["sphere", "cube", "mug", "pin", "dualshock"]}; print("Using Baxter")
elif args.environment == "kuka":
    ENVS = {obj:gym.make('kuka_grasping-v0', display=False, obj=obj) for obj in ["mug"]}; print("Using Kuka")

def extract_individuals(folder, select: Union[int, str], keep: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    folder_path = Path(folder)
    out = []
    runs = list(folder_path.glob("**/run_details.yaml"))
    for run in runs:
        with open(run, "r") as f:
            d = yaml.safe_load(f)
        d['object'] = d["env kwargs"]["obj"] # alias
        if keep is not None:
            for k, v in keep.items():
                if (isinstance(v, (list, tuple)) and d[k] not in v) or d[k] != v:
                    continue
        if d["env kwargs"]["id"][:-12] != args.environment or not d['successful']: continue # skip
        is_robustness = d["multi quality"] is not None and '+grasp robustness' in {q for qs in d["multi quality"] for q in qs}
        repertoire = np.load(run.parent/"individuals.npz")
        if isinstance(select, int):
            individuals = repertoire["genotypes"][:select]
        elif select == "transfer":
            individuals = repertoire["genotypes"][repertoire['transfer samples']]
        elif select == "diversity":
            individuals = repertoire["diversity samples"]
        elif select == "genotype":
            individuals = repertoire["genotype samples"]
        else:
            print("select must be in {'all', 'diversity', 'genotype', 'transfer'}."); return
        for i, ind in enumerate(individuals):
            out.append({
                "ind": ind,
                "object": d["env kwargs"]["obj"],
                "robustness": is_robustness,
                "controller info": d['controller info'],
                "robot": d["env kwargs"]["id"],
                "name": f"{run.parent.name}_{repertoire['transfer samples'][i] if select == 'transfer' else i}"
            })
    return out

def evaluate_robustness(folder, repeat=10, linspace={"start":0, "stop":0.05, "num":11}, select=20, keep=None):
    """ Evaluate individuals with different noises. """
    folder_path = Path(folder)
    df = pd.DataFrame(columns=["robot", "object", "robustness", "σ", "success rate", "controller info", "ind"])
    linspace['start'] = linspace.get('start', 0)
    σs = np.linspace(**linspace)

    data = extract_individuals(folder, select, keep)
    for σ in σs:
        df_ = pd.DataFrame(data)
        df_.loc[:, "σ"] = σ
        df = pd.concat(ignore_index=True, objs=[df, df_])

    if len(df)==0: sys.exit("No individual found")

    dpos = {σ.item():np.random.normal(scale=σ, size=(repeat,2)) for σ in σs}
    #print(list(df['σ']))
    args = [{'delta pos': dpos[σ], "controller info": c, "ind": ind, 'obj':obj} for σ, c, ind, obj in zip(df['σ'], df['controller info'], df['ind'], df["object"])]

    with Pool() as p:
        df = df.assign(**{"success rate": np.array(p.map(simulate1, args))})

    df = df[["robot", "object", "robustness", "σ", "success rate"]] # keep essentials
    df.to_csv(folder_path/"evaluateWithNoise.csv")

    g = sns.relplot(x="σ", y="success rate", hue="robustness", data=df, kind="line", height=3, aspect=1.5)
    g._legend.set_bbox_to_anchor([0.8,0.8])
    g.savefig(folder_path/"evaluateWithNoise.pdf")




def simulate1(data):
    delta_pos, controller_info, ind, obj = data['delta pos'], data['controller info'], data['ind'], data['obj']
    global ENVS
    env = ENVS[obj]
    def sim(delta_pos):
        o = env.reset(delta_pos=delta_pos, load='state')
        controller = InterpolateKeyPointsGrip(ind, **controller_info, initial=env.get_joint_state())
        touch = False
        for k in range(controller_info['n_iter']): # simulation
            o, r, eo, inf = env.step(controller.get_action(k,o))
            touch = touch or len(inf['contact object robot'])>0
        return o, r, eo, inf, touch

    #if not sim([0,0])[-1]: return 0
    successRatio = 0
    for dp in delta_pos:
        o, r, eo, inf, touch = sim(delta_pos=dp)
        successRatio += r
    return successRatio / len(delta_pos)


def evaluate_robustness_object(folder, r=0.02, repeat=20, select="transfer", keep=None):
    """ Evaluate individuals with differents objects. """
    folder_path = Path(folder)

    df = pd.DataFrame(extract_individuals(folder, select, keep)).set_index('name')
    df.loc[:, "simulation grasping success rate"] = 0


    if len(df)==0: sys.exit("No individual found")

    # random delta_pos
    delta_pos = np.zeros((repeat, 2))
    for i in range(repeat):
        dist = float("inf")
        while dist > r:
            delta_pos[i] = (np.random.rand(2)*2-1)*r
            dist = np.linalg.norm(delta_pos[i])
    #delta_pos = np.arange(repeat)/repeat*2*np.pi
    #delta_pos = np.vstack([np.cos(rangeRepeat), np.sin(rangeRepeat)]).T*r

    data = [{'delta pos': delta_pos, "controller info": c, "ind": ind, 'obj':obj} for c, ind, obj in zip(df['controller info'], df['ind'],df["object"])]

    #with Pool() as p:
    df = df.assign(**{"simulation grasping success rate": np.array(futures.map(simulate1, data))})

    df = df.drop(columns=["ind", "controller info"])
    df.to_csv(folder_path/"evaluateWithNoiseObject.csv")
    snsplot = sns.catplot(x="object", y="simulation grasping success rate", hue="robustness", data=df, kind="box")
    snsplot.savefig(folder_path/"evaluateWithNoiseObject.pdf")


def simulate2(controller_info, ind, obj="pin"):
    global ENVS
    env = ENVS[obj]
    o = env.reset(load='all')
    #env.p.stepSimulation()
    controller = InterpolateKeyPointsGrip(ind, **controller_info, initial=env.get_joint_state())
    touch = False
    maxTorque = 0
    for k in range(controller_info['n_iter']): # simulation
        #action = controller.get_action(k)
        o, r, eo, inf = env.step(controller.get_action(k, o))
        touch = touch or len(inf['contact robot table'])>0
        maxTorque = np.maximum(maxTorque, np.abs(inf['applied joint motor torques'][:-env.n_control_gripper]).sum())
    return touch, maxTorque, r


def contactTable(folder, outFolder):
    folderPath = Path(folder)
    df = pd.DataFrame(columns=["robot","object","quality","energy","touch","max torque"])
    runs = folderPath.glob("**/run_details.*")# ; runs = list(runs)[:10]
    for run in runs:
        with open(run, "r") as f:
            d = json.load(f) if run.suffixes[0] == '.json' else yaml.safe_load(f)
        individuals = run.parent.glob("*.npy")
        for indPath in individuals:
            if run.suffixes[0] == '.json':
                df = df.append(pd.Series([d["robot"], d["object"], d["multi quality"] is not None, "Never", False, 0], index=df.columns, name=indPath))
            else:
                df = df.append(pd.Series([d["robot"], d["object"], '+grasp robustness' in d["multi quality"][1], "Always" if '-energy' in d["multi quality"][1] else '1-4 only', False, 0], index=df.columns, name=indPath))

    if len(df)==0: sys.exit("No individual found")

    robot_id = ROBOT_ID[d["robot"]] # convert string to int
    env_name, gene_per_keypoints, nb_step_to_rollout, nb_iter = ENV_NAME[robot_id], GENE_PER_KEYPOINTS[robot_id], NB_STEPS_TO_ROLLOUT[robot_id], NB_ITER[robot_id]
    controller_info = {'pause_frac': 0.66, 'n_iter': nb_iter, 'NB_KEYPOINTS': 3, 'GENE_PER_KEYPOINTS': gene_per_keypoints}
    simulateWrapper = partial(simulate2, controller_info)

    with Pool() as p:
        touch, maxTorque, r = np.array(p.starmap(simulate2, zip(map(np.load, df.index),df["object"]))).T
        df = df.assign(**{"touch":touch, 'max torque':maxTorque})

    outPath = Path(outFolder) if outFolder else Path()
    df.to_csv(outPath/"contactTable.csv")
    fig, ax = plt.subplots(2,3, figsize=(20,15))
    objects = df['object'].unique()
    for i, energy in enumerate(('Never', 'Always', '1-4 only')):
        ax[1,i].set_title('energy: '+ energy)
        if len(df[df['energy']=='Never'])>0:
            sns.barplot(x="object", y='touch', hue="quality", data=df[df['energy']==energy], ax=ax[0,i], order=objects)
            sns.boxplot(x="object", y="max torque", hue="quality", data=df[df['energy']==energy], ax=ax[1,i], order=objects)
    fig.savefig(outPath/"contactTable.pdf")

def success(folder, csvPath=None):
    folderPath = Path(folder)
    if csvPath is None:
        df = pd.DataFrame(columns=["robot", "object", "robustness", "success", "max torque", "controller info"])
        runs = folderPath.glob("**/run_details.yaml")# ; runs = list(runs)[:10]
        for run in runs:
            with open(run, "r") as f:
                d = yaml.safe_load(f)
            if "env kwargs" not in d or d["env kwargs"]["id"] != 'kuka_grasping-v0': continue
            is_robustness = d["multi quality"] is not None and '+grasp robustness' in {q for qs in d["multi quality"] for q in qs}
            individuals = run.parent.glob("*.npy")
            for indPath in individuals:
                df = df.append(pd.Series([d["env kwargs"]["id"], d["env kwargs"]["obj"], is_robustness, False, 0, d["controller info"]], index=df.columns, name=indPath))

        if len(df)==0: sys.exit("No individual found")

        with Pool() as p:
            touch, maxTorque, r = np.array(p.starmap(simulate2, zip(df['controller info'], map(np.load, df.index), df["object"]))).T
            df = df.assign(**{"success":r, "max torque":maxTorque})
        df.to_csv(folderPath/"success.csv")
    else:
        df = pd.read_csv(csvPath)

    df = df.groupby(["robot", "object", "robustness"]).mean().reset_index(inplace=True)

    sns.catplot(x="object", y="success", hue="robustness", data=df, kind='bar').savefig(folderPath/"success.pdf")

def evalEnergy(individuals, outFolder):
    fig, ax  = plt.subplots(2,1)
    for label, ind in individuals.items():
        with open(Path(ind).parent/"run_details.yaml", 'r') as f:
            d = yaml.safe_load(f)
        ind = np.load(ind)
        env = ENVS[d['object']]

        env.reset()
        controller = InterpolateKeyPointsEndPauseGripAssumption(ind, d['controller info'], initial=env.get_joint_state())
        action = controller.initial_action
        torque = np.zeros(d['controller info']['n_iter'])
        for k in range(d['controller info']['n_iter']): # simulation
            o, r, eo, inf = env.step(controller.get_action(k))
            torque[k] = np.abs(inf['applied joint motor torques']).sum()
        ax[0].plot(torque, label=label)
        ax[1].plot(np.cumsum(torque), label=label)
    ax[0].legend()
    ax[1].legend()
    fig.suptitle("Sum of joint torques and the cumulative")
    fig.savefig(outFolder+"/energy.pdf")

def torqueThreshold(csvPath, thresholds=[150,200]):
    df = pd.read_csv(csvPath)
    dfRate = pd.DataFrame(columns=["object","quality","energy","rate", "threshold"])
    for energy in df['energy'].unique():
        for object in df['object'].unique():
            for quality in df['quality'].unique():
                for threshold in thresholds:
                    mask = (df['energy']==energy) & (df['object']==object) & (df['quality']==quality)
                    count = np.count_nonzero(mask)
                    if count > 0:
                        rate = np.count_nonzero(mask & (df['max torque']<threshold)) / count
                        dfRate = dfRate.append({"object":object,"quality":quality,"energy":energy,"rate":rate, "threshold":threshold}, ignore_index=True)

    snsplot = sns.catplot(x="object", y="rate", hue="quality", col="energy", row="threshold", kind="bar", height=4, aspect=1.5, data=dfRate)
    snsplot.savefig('thresholdRate.pdf')

if __name__ == '__main__':
    evaluation, kwargs = {
        'noise': (evaluate_robustness, {'folder':args.runs, "repeat": args.n, "linspace": {"stop":args.σ, "num":args.intervals}, "select": args.select, "keep": args.keep}),
        'object': (evaluate_robustness_object, {'folder':args.runs, "r": args.σ, "repeat": args.n, "select": args.select, "keep": args.keep}),
    }[args.evaluation]
    evaluation(**kwargs)
