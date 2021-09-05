#!/usr/bin/env python3
# plot the results of run_details.yaml
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from pathlib import Path
from collections.abc import Sized
from operator import and_, or_
from functools import reduce
import argparse
import json, yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering

import seaborn as sns
sns.set_theme(style="whitegrid")

def plot_details(folder, x, y, hue=None, keep=None, figsize:tuple=None, order=None, xticklabels=None, kind=None, rotate_xticklabels=False, set_fig=None, ):
	columns = ["evaluation function", "nb of generations", "pop size", "archive limit size", "archive limit strat", "robustness", "controller", "object", "robot", "run time", "diversity coverage", "diversity uniformity", "number of successful", "number of clusters", "sample efficiency", "algo type", "quality", "first success generation", "successful"]
	assert x in columns, f"x={x} must be in {columns}"
	if hue is not None:
		assert hue in columns, f"hue={hue} must be in {columns}"
	if isinstance(y, str):
		assert y in columns, f"y={y} must be in {columns}"
	else:
		for i, metric in enumerate(y):
			assert metric in columns, f"y[i]={metric} must be in {columns}"

	folder_path = Path(folder)
	details = folder_path.glob("**/run_details.yaml")
	df = pd.DataFrame(columns=columns)

	for i, detail in enumerate(details):
		with open(detail, "r") as f:
			d = yaml.safe_load(f)
		qual = d.get('multi quality', None)
		qualities = {q for qs in qual or {} for q in qs}
		d['quality'] = isinstance(qual, str) or (isinstance(qual, list) and len(qualities)>0)
		d["robustness"] = '+grasp robustness' in qualities
		d["sample efficiency"] = d["number of successful"] / d["n evaluations"]
		if 'env kwargs' not in d: continue
		d['robot'] = d["robot"]#d['env kwargs']['id'][:-12]
		d['object'] = d['env kwargs']['obj']
		d['diversity coverage'] = d.get('diversity coverage', 0)
		if d['algo type'] == 'ns_rand_multi_bd':
			d['algo type'] = {'pos_div_pos_grip':'4BD', 'pos_div_pos':'NSMBS no BD 2', 'pos_div_grip':'NSMBS no BD 3', 'random_search': 'random', 'map_elites': 'map-elites', 'ns_nov': 'NS'}[d['behaviour descriptor']]
		elif d['algo type'] in {'random_search', 'ns_nov', 'map_elites'}:
			d['algo type'] = {'random_search': 'random', 'ns_nov': 'NS', 'map_elites':'map-elites'}[d['algo type']]



		df = df.append({c:d.get(c, np.NaN) if c in d.keys() else np.NaN for c in df.columns}, ignore_index=True)
	if len(df)==0:
		print("No run found"); return

	if keep is not None and len(keep)>0:
		for key, value in keep.items():
			df = df[reduce(or_, [df[key]==v for v in value]) if isinstance(value, (list, tuple, set, np.ndarray)) else df[key]==value]
	#print(df[df['algo type']=="4BD"])

	df = df.astype({"first success generation": float,"quality": bool, "robustness": bool, "number of clusters": int, "sample efficiency": float, "diversity coverage": float, "number of successful":int, "successful": int})
	#print(df.groupby(['algo type'])["number of successful"].mean()) ;print(df.groupby(['algo type'])["number of successful"].std()); exit()

	if isinstance(y, str):
		#if figsize is not None: sns.set(rc={"figure.figsize":figsize})
		height = 5 if figsize is None else figsize[1]
		aspect = 1 if figsize is None else figsize[0]/figsize[1]
		plot_kwargs = {'violin': {"scale": "width"}, "box":{}, 'bar':{'ci':None}, 'strip':{}}[kind]

		fig = sns.catplot(x=x, y=y, hue=hue, data=df, order=order, kind=kind, height=height, aspect=aspect, **plot_kwargs)
		if xticklabels is not None: fig.axes.flat[0].set_xticklabels(xticklabels)
		if rotate_xticklabels:
			fig.set_xticklabels(fig.axes.flat[0].get_xticklabels(), rotation=30, horizontalalignment='right') # rotation
		if set_fig is not None:
			fig.set(**set_fig)
		fig.savefig(folder_path/f"{x}_{y}_{kind}.pdf".replace(' ', '_'), bbox_inches = 'tight')

	else:
		ncol = min(2, len(y)) # nb of columns
		nrow = np.ceil(len(y)/ncol).astype(int)
		fig, ax = plt.subplots(nrow, ncol, figsize=figsize)
		ax = np.array(ax).reshape(nrow, ncol) # make it 2D
		xticklabels_kwargs = {'rotation':30, 'horizontalalignment':'right'} if rotate_xticklabels else {}
		for i, metric in enumerate(y):
			g = sns.violinplot(x=x, y=metric, hue=hue, data=df, ax=ax[i//ncol, i%ncol], order=order, scale='width')
			g = sns.stripplot(x=x, y=metric, hue=hue, data=df, order=order, color='.4', ax=ax[i//ncol, i%ncol], size=4, jitter=0.2)

			if xticklabels is not None:
				ax[i//ncol, i%ncol].set_xticks(range(len(df[x].unique())))
				ax[i//ncol, i%ncol].set_xticklabels(xticklabels, **xticklabels_kwargs)
			if set_fig is not None:
				g.set(**set_fig)
		fig.savefig(folder_path/f"{x}_{'_'.join(y)}.pdf".replace(' ', '_'), bbox_inches = 'tight')




def eligibilityRates(path, obj='cube'):
	folder = Path(path)
	runs = list(folder.glob('**/run_details.json'))
	if len(runs)==0:
		print("No run found"); return
	run = runs[0]
	with open(run, "r") as f:
		d = json.load(f)
	ngen = d['nb of generations']
	nbd = len(np.unique(d['bd indexes']))
	eligibility = np.zeros((ngen, nbd))
	i = 0
	for run in runs:
		with open(run, "r") as f:
			d = json.load(f)
		if d['object'] != obj: continue
		with open(run.parent/"run_data.json", "r") as f:
			data = json.load(f)
		assert len(data['eligibility rates']) == ngen and len(data['eligibility rates'][0]) == nbd, 'The shape of the eligibility rates array is not consistent'
		eligibility += data['eligibility rates']
		i += 1
	if i>0: eligibility /= i
	eligibility = eligibility[:300]
	fig, ax = plt.subplots(1,1)
	ax.plot(eligibility[:,0], label="Always")
	ax.plot(eligibility[:,3], label="Contact")
	ax.plot(eligibility[:,2], label="Grasp")
	ax.set_title(f'Average eligibility on {i} runs')
	ax.set_ylabel('Eligibility rate')
	ax.set_xlabel('Generation')
	ax.legend()
	plt.show()

def tsne_robot(runs: List[str], n_randoms=10000):
	""" Reduce the diversity data (position or orientation) of all individuals for 1 run to 2D with t-sne.
		runs: : runs to analyze
	"""
	runs_path = [Path(path) for path in runs]
	df = pd.DataFrame(columns=["robot", "x", "y", "cluster"])
	rng = np.random.default_rng()
	metric = lambda a,b: np.min(np.linalg.norm([[a+b],[a-b]], axis=-1))
	for folder in runs:
		folder = Path(folder)
		with open(folder/"run_details.yaml", "r") as f:
			d = yaml.safe_load(f)
		data_inds = np.load(folder/"individuals.npz")
		n = len(data_inds["genotypes"])
		random_quaternion = rng.random(size=(n,4))*2-1
		random_quaternion /= np.linalg.norm(random_quaternion, axis=-1)[:,None]
		q = np.vstack((data_inds["wxyzs"], random_quaternion))
		distance_matrix = np.min(np.linalg.norm([q[:-n]+q[:-n,None], q[:-n]-q[:-n,None]], axis=-1), axis=0)
		clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', compute_full_tree=True, distance_threshold=0.4, linkage='average')
		clustering = clustering.fit(distance_matrix)

		xys = TSNE(n_components=2, n_jobs=-1, metric=metric).fit_transform(q)[:-n]
		df = pd.concat([df, pd.DataFrame({"robot":d['env kwargs']['id'][:-12], "x":xys[:,0], "y":xys[:,1], "cluster":list(map(str,clustering.labels_))})])
	g = sns.relplot(x="x", y="y", hue="cluster", col="robot", data=df, kind="scatter", s=2, edgecolors='none', linewidths=0, height=4, aspect=1, alpha=0.4)
	g.savefig("orientation-diversity.pdf")
	#plt.show()

def tsne_runs(folder, n_randoms=10000):
	""" Reduce the diversity orientation of all individuals for all runs to 2D with t-sne.
		runs: runs to analyze
	"""
	rng = np.random.default_rng()
	folder_path = Path(folder)
	df = pd.DataFrame(columns=["run", "x", "y"])
	diversities = []
	nb_success_per_run = []
	for run in folder_path.glob("**/run_details.yaml"):
		with open(run, "r") as f:
			d = yaml.safe_load(f)
		if d['algo type'] != 'ns_rand_multi_bd' or d['behaviour descriptor'] != 'pos_div_pos_grip' or (d['multi quality'] is not None and '+grasp robustness' in {q for qs in d['multi quality'] for q in qs}) or not d['successful']: continue # skip

		data_inds = np.load(run.parent/"individuals.npz")
		diversities.append(data_inds["wxyzs"])
		nb_success_per_run.append(len(data_inds["genotypes"]))

	random_quaternion = rng.random(size=(n_randoms,4))*2-1
	random_quaternion /= np.linalg.norm(random_quaternion, axis=-1)[:,None]
	diversities.append(random_quaternion)

	q = np.vstack(diversities)
	#distance_matrix = np.minimum(np.linalg.norm(q+q[:,None], axis=-1), np.linalg.norm(q-q[:,None], axis=-1))

	xys = TSNE(n_components=2, n_jobs=-1, metric=lambda a,b: np.min(np.linalg.norm([[a+b],[a-b]], axis=-1))).fit_transform(q)[:-n_randoms]
	for i, xys_per_run in enumerate(np.split(xys, np.cumsum(nb_success_per_run[:-1]))):
		df = pd.concat([df, pd.DataFrame({'run':str(i), 'x':xys_per_run[:,0], 'y':xys_per_run[:,1]})])
	g = sns.relplot(x="x", y="y", hue="run", data=df, kind="scatter", s=2, edgecolors='none', linewidths=0, height=4, aspect=1, alpha=0.4)
	g.savefig(folder_path/"runs-diversity.pdf")
	#plt.show()

def plot_hist(folder, y, hue, keep=None, labels=None, order=None):
	folder_path = Path(folder)
	df = pd.DataFrame(columns=["robot", "algorithm", "generation", "sample efficiency", "robustness", "n clusters", "energy"])
	for run in Path(folder).glob("**/run_details.yaml"):
		with open(run, "r") as f:
			d = yaml.safe_load(f)
		if 'env kwargs' not in d: continue
		qual = d.get('multi quality', None)
		qualities = {q for qs in qual or {} for q in qs}
		d['quality'] = isinstance(qual, str) or (isinstance(qual, list) and len(qualities)>0)
		if d['algo type'] == 'ns_rand_multi_bd':
			d['algo type'] = {'pos_div_pos_grip':'4BD', 'pos_div_pos':'NSMBS no BD 2', 'pos_div_grip':'NSMBS no BD 3', 'random_search': 'random', 'map_elites': 'map-elites', 'ns_nov': 'NS'}[d['behaviour descriptor']]
		elif d['algo type'] in {'random_search', 'ns_nov'}:
			d['algo type'] = {'random_search': 'random', 'ns_nov': 'NS'}[d['algo type']]
		hist = np.load(run.parent/"run_data.npz")
		#print(list(hist.keys()))
		if 'n clusters' not in hist or 'evaluations' not in hist: continue
		# clusters
		n_clusters = np.zeros(d['nb of generations'])
		n_clusters[:] = np.nan
		inter = int(d['nb of generations'] / len(hist['n clusters']))
		n_clusters[0::inter] = hist['n clusters']

		df = pd.concat(ignore_index=True, objs=[df, pd.DataFrame({
			'robot': d['robot'],
			'object': d['env kwargs']['obj'],
			'sample efficiency': np.cumsum(hist['successes']) / np.cumsum(hist['evaluations']),
			'sample efficiency repetitions': np.cumsum(hist['successes']) / np.cumsum(hist['evaluations including repetitions']), # including
			'generation': range(d['nb of generations']),
			'algorithm': d['algo type'],
			'robustness': d['multi quality'] is not None and '+grasp robustness' in {q for qs in d['multi quality'] for q in qs},
			'n clusters': n_clusters,
			'energy': hist.get('energy', hist.get('-energy', np.nan))
		})])

	if len(df) == 0:
		print("no individual found")
		return

	if keep is not None and len(keep)>0:
		for key, value in keep.items():
			df = df[reduce(or_, [df[key]==v for v in value]) if isinstance(value, (list, tuple, set, np.ndarray)) else df[key]==value]

	g = sns.relplot(data=df, x="generation", y=y, hue=hue, kind="line", hue_order=order, height=3, aspect=1.5)
	if labels is not None:
		for t,l in zip(g._legend.texts, labels): t.set_text(l)
	#g._legend.set_bbox_to_anchor([1.1,0.5])
	g.savefig(folder_path/f"{hue}_{y}_per_generation.pdf".replace(' ', '_'))
	#plt.show()


def plot_grasping_style(csv_file, xticklabels=None, order=None):
	df = pd.read_csv(csv_file).drop(columns="run")

	# normalize per robot
	df = df.groupby(by=["robot"]).sum()
	df = df.div(df.sum(axis=1), axis=0).reset_index(level=0) # normalize to get proportion

	df = pd.melt(df, id_vars=["robot"], value_vars=["handle", "in", "out", "in-out"], var_name="style", value_name="proportion")
	g = sns.catplot(data=df, x="robot", y="proportion", hue="style", kind="bar", height=3, aspect=1.5, order=order)
	g._legend.set_bbox_to_anchor([0.4,0.68])
	g.set(xlabel="")
	if xticklabels is not None: g.axes.flat[0].set_xticklabels(xticklabels)
	g.savefig(Path(csv_file).parent/"mug_grasping_style.pdf")


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-r", "--runs", help="The directory containing runs", type=str, default=str(Path(__file__).parent.parent/'runs'))
	args = parser.parse_args()

	robot_order = ["kuka", "baxter", "pepper"]
	algo_order =  ["random", "map-elites", "NS", "NSMBS no BD 2", "NSMBS no BD 3", "4BD"] #['multiBDSel+quality', '4BD', 'nsmbs_optimal', 'dynamic programming']
	labels = ["random", "map-elites", "NS", "NSMBS no BD 2", "NSMBS no BD 3", "NSMBS"] #['multiBDSel+quality', 'multiBDSel', 'n=100,multiBDSel', 'ε=0.5, dynamic programming']
	fig_size = (4,3)
	kind = "strip"
	folder = args.runs
	for y, ylabel in {"number of successful": "number of solutions", "successful": "successful run rate", "first success generation": None, "sample efficiency": None, "diversity coverage": None}.items():
		#break
		plot_details(
			folder,
			x="algo type",
			y=y if y=="successful" else [y],
			hue=None,
			keep={'algo type':("random", "map-elites", "NS", "NSMBS no BD 2", "NSMBS no BD 3", "4BD"), 'robot':'baxter', 'robustness':False, 'object':'mug'},
			order=algo_order,
			xticklabels=labels,
			figsize=fig_size,
			rotate_xticklabels=True,
			set_fig={"xlabel":None} if ylabel is None else {'ylabel': ylabel, "xlabel":None},
			kind="bar" if y=="successful" else kind,
		)

	#for y in {"diversity coverage", "sample efficiency"}:
		#plot_details(folder, x="robot", y=[y], hue=None, keep={'algo type':'4BD', 'robustness': False, 'object':'mug'}, order=robot_order, figsize=fig_size, kind=kind, set_fig={'xlabel':None})
	plot_hist(folder, y='sample efficiency', hue="algorithm", order=algo_order, labels=labels, keep={'algorithm':("random", "map-elites", "NS", "NSMBS no BD 2", "NSMBS no BD 3", "4BD"), 'robot':'baxter', 'robustness':False, 'object':'mug'})
	#plot_hist(folder, y='sample efficiency', hue="object", order=["cube", "dualshock", "mug", "pin", "sphere"], keep={'algorithm':"4BD", 'robot':'baxter', 'robustness':False})
	#plot_hist(folder, y='energy', hue="algorithm", order=algo_order, labels=labels)
	#plot_hist(folder, y="sample efficiency", hue="robot", keep={'algorithm':"4BD", "robustness":False, 'object':'mug'}, order=robot_order)
	#plot_grasping_style(Path(folder).parent/'mug.csv', order=robot_order)
