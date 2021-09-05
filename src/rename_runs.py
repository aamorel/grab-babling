import argsparse
from pathlib import Path
import json, yaml

import numpy as np


def rename_runs(folder):
	""" rename run folders indicating the object and the robustness. """
	folderPath = Path(folder)
	runs = folderPath.glob("**/run_details.yaml")
	countName = dict()
	for run in [r for r in runs]:
		with open(run) as f:
			d = yaml.safe_load(f)
		is_robustness = d["multi quality"] is not None and '+grasp robustness' in {q for qs in d["multi quality"] for q in qs}
		key = f"{d['env kwargs']['obj']}{'RobustnessYes' if is_robustness else 'RobustnessNo'}"
		if key in countName.keys():
			countName[key] += 1
		else:
			countName[key] = 0
		while (folderPath/f"{key}{countName[key]}").exists():
			countName[key] += 1
		run.parent.rename(folderPath/f"{key}{countName[key]}")



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-r", "--runs", help="The directory containing runs", type=str, default=str(Path(__file__).parent.parent/'runs'))
	args = parser.parse_args()
	rename_runs(args.runs)
