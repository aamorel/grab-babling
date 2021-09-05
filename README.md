## Requirements
Create a Python virtual environment e.g. `python3 -m venv venv` and activate it `source venv/bin/activate`.
Clone the repositories `git clone -b yakumo https://github.com/aamorel/grab-babling.git && git clone https://github.com/Yakumoo/sbil.git` then `cd grab-babling`.
Install dependencies `pip install -r grab-babling/requirements.txt && pip install -e grab-babling/gym_grabbing sbil`.

## Package descriptions
`grab-babling/gym_grabbing` is the gym package for robot environments.
`grab-babling/src` contains scripts for QD search.
`sbil` is an imitation RL library.

## NSMBS, Novelty Search with Multiple Behaviour Spaces
`cd grab-babling && mkdir -p runs`: go to the right directory and make sure `runs` folder exists.
`python -m scoop src/applynoveltygrasping.py -r kuka -m 'joint velocities' -o cube` will launch a NSMBS search and results are logged in `runs`.
The default mode is joint positions and it works fine.

## Generate demo_buffer
`cd grab-babling/rl && python generateBuffer.py` will create a ReplayBuffer filled with demonstrations in `data` folder.

## Reinforcement learning from demonstrations
Create an appropriate yaml config file. An example is given [here](https://github.com/aamorel/grab-babling/blob/yakumo/rl/kuka.yaml).
Then learn: `python -m sbil.learn -c path/config.yaml`
