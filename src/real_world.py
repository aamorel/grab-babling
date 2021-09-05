
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json, yaml
import pandas as pd
from functools import reduce, partial

sns.set_theme(style="whitegrid")

def plot_success(success_folder, runs_folder): # success on the real robot baxter
	runs_folder = Path(runs_folder)
	df = pd.DataFrame(columns=["object", "robustness", "success"])

	for succes_file in Path(success_folder).glob("*.json"):
		with open(succes_file, "r") as f:
			s = json.load(f)
		run = str(Path(s['order'][0]).stem).split('_')[0]
		df = pd.concat(ignore_index=True, objs=[df, pd.DataFrame({
			"run": run,
			"object": "",
			"robustness": False,
			"success": [value for key,value in s.items() if isinstance(value, (float))],
		})])
	# find failed runs to add in df to compute sample efficiency

	df = df.groupby(["run", "object", "robustness"]).mean().reset_index(level=['object', "robustness"]) # mean success over run, leave run as index
	df_runs = df.index
	for run in runs_folder.glob("**/run_details.yaml"):
		with open(run, "r") as f:
			d = yaml.safe_load(f)
		run_name = str(run.parent.name)
		if run_name not in df_runs:
			df.append(ignore_index=True, other={"run": run_name, "object": d['env kwargs']['obj'], "robustness":False, "success":-1, "sample efficiency":0})
		else:
			df.loc[run_name, "sample efficiency"] = df.loc[run_name, "success"] * d['number of successful'] / d['n evaluations']
			df.loc[run_name, "object"] = d['env kwargs']['obj']
			df.loc[run_name, "robustness"] = isinstance(d['multi quality'], list) and '+grasp robustness' in {q for qs in d['multi quality'] for q in qs}

	df = df.reset_index(level=0)
	order = ["cube", "dualshock", "mug", "pin", "sphere"]
	g = sns.catplot(data=df[df['success']>=0], x="object", y="success", hue=None, kind="bar", height=4, aspect=1, order=order)
	#g._legend.set_bbox_to_anchor([0.5,0.79])
	g.set(xlabel=None)
	g.savefig(success_folder+"/success_rate.pdf")

	g = sns.catplot(data=df, x="object", y="sample efficiency", hue=None, kind="bar", height=4, aspect=1, legend=True, order=order)
	#g._legend.set_bbox_to_anchor([0.7,0.8])
	g.set(xlabel=None)
	g.savefig(success_folder+"/sample_efficiency.pdf")

def robustness_success(successes_folder: str, robustness_csv: str):
	df = pd.read_csv(robustness_csv); df = df.set_index('Unnamed: 0') # change index
	df = df.assign(**{"reality grasping success rate": 0})# add column
	#df = pd.DataFrame(columns=[ "ind", "object", "robustness", "reality grasping success rate", "simulation grasping success rate"])
	for folder in Path(successes_folder).glob("*.json"):
		with open(folder, "r") as f:
			info = json.load(f)
		for k, v in info.items():
			if isinstance(v, float):
				ind = k.split("/")[-1].split('.')[0]
				df.loc[ind, "reality grasping success rate"] = v

	df = df.assign(**{"reality grasping success": lambda x: x["reality grasping success rate"]>0})
	g = sns.catplot(y="simulation grasping success rate", x="object", hue="reality grasping success", data=df, kind="box", height=3, aspect=1.5, order=["cube", "dualshock", "mug", "pin", "sphere"])
	g.set(xlabel=None)
	g._legend.set_title("Reality\ngrasping\nsuccess")
	g._legend.set_bbox_to_anchor([0.85,0.5])
	#g._legend.set_bbox_to_anchor([0.65,0.26])#[0.85,0.5])
	g.savefig(successes_folder+"/robustness_success_correlation2.pdf")

if __name__ == '__main__':
	#robustness_success('/Users/Yakumo/Downloads/exp0/success', '/Users/Yakumo/Downloads/exp0/baxter/evaluateWithNoiseObject.csv')
	plot_success("/Users/Yakumo/Downloads/exp3/baxter_successes", "/Users/Yakumo/Downloads/exp3/baxter_wo_quality_transfer")
