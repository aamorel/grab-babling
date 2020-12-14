import gym
import slimevolleygym  # must still be imported
import noveltysearch
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from deap import base, creator

DISPLAY = False
PARALLELIZE = True
GEN = 100
POP_SIZE = 100
ARCHIVE_LIMIT = None
NB_CELLS = 100
BD_BOUNDS = [[0, 1], [0, 1]]
INITIAL_GENOTYPE_SIZE = 99
N_EXP = 30
COV_TYPE = 'coverage'  # 'coverage' or 'archive coverage'
UNI_TYPE = 'uniformity'  # 'uniformity' or 'archive uniformity'

ALGO = 'ns_rand'
PLOT = True
ARCHIVE_ANALYSIS = False

if PARALLELIZE:
    # container for behavior descriptor
    creator.create('BehaviorDescriptor', list)
    # container for info
    creator.create('Info', dict)
    # container for genetic info
    creator.create('GenInfo', dict)
    # container for novelty
    creator.create('Novelty', base.Fitness, weights=(1.0,))
    # container for fitness
    if min:
        creator.create('Fit', base.Fitness, weights=(-1.0,))
    else:
        creator.create('Fit', base.Fitness, weights=(1.0,))

    # container for individual
    creator.create('Individual', list, behavior_descriptor=creator.BehaviorDescriptor,
                   novelty=creator.Novelty, fitness=creator.Fit, info=creator.Info,
                   gen_info=creator.GenInfo)

    # set creator
    noveltysearch.set_creator(creator)


class ControllerNet(nn.Module):

    def __init__(self, params):
        super(ControllerNet, self).__init__()
        self.l1 = nn.Linear(12, 6)
        weight_1 = nn.Parameter(torch.Tensor(params[:72]).reshape((6, 12)))
        bias_1 = nn.Parameter(torch.Tensor(params[72:78]).reshape(6))
        self.l1.weight = weight_1
        self.l1.bias = bias_1
        self.l2 = nn.Linear(6, 3)
        weight_2 = nn.Parameter(torch.Tensor(params[78:96]).reshape((3, 6)))
        bias_2 = nn.Parameter(torch.Tensor(params[96:99]).reshape(3))
        self.l2.weight = weight_2
        self.l2.bias = bias_2
        self.r1 = nn.ReLU()
        self.r2 = nn.Sigmoid()

    def forward(self, ind):
        ind = torch.Tensor(ind)
        ind = self.r1(self.l1(ind))
        action = self.r2(self.l2(ind)).numpy()
        
        action[0] = int(action[0] > 0.5)
        action[1] = int(action[1] > 0.5)
        action[2] = int(action[2] > 0.5)

        return action


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def evaluate_individual(individual):
    """Evaluates an individual: computes its value in the behavior descriptor space,
    and its fitness value.

    Args:
        individual (Individual): an individual

    Returns:
        tuple: tuple of behavior (list) and fitness(tuple)
    """
    inf = {}

    env = gym.make("SlimeVolley-v0")
    env.reset()

    action = [0, 0, 0]
    eo = False
    count = 0
    reward = 0
    distance = 0

    # CONTROLLER
    individual = np.array(individual)
    controller = ControllerNet(individual)
    while not eo:
        count += 1
        if DISPLAY:
            env.render()
        # apply previously chosen action
        o, r, eo, info = env.step(action)
        reward += r
        with torch.no_grad():
            action = controller.forward(o)

        dist_to_ball = math.sqrt((o[0] - o[4])**2 + (o[1] - o[5])**2)
        distance += dist_to_ball

        if(DISPLAY):
            time.sleep(0.01)

    # use last info to compute behavior and fitness
    mean_distance = distance / (count * 4)
    behavior = [count / 3000, mean_distance]
    fitness = reward
    return (behavior, (fitness,), inf)


if __name__ == "__main__":
    if not ARCHIVE_ANALYSIS:

        pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE, BD_BOUNDS,
                                                             mini=False, archive_limit_size=ARCHIVE_LIMIT,
                                                             plot=PLOT, algo_type=ALGO, nb_gen=GEN,
                                                             parallelize=PARALLELIZE,
                                                             measures=True, pop_size=POP_SIZE, nb_cells=NB_CELLS)

        if PLOT:
            # plot final states

            archive_behavior = np.array([ind.behavior_descriptor.values for ind in archive])
            pop_behavior = np.array([ind.behavior_descriptor.values for ind in pop])
            hof_behavior = np.array([ind.behavior_descriptor.values for ind in hof])
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set(title='Final Archive', xlabel='Game duration', ylabel='Mean distance between player and ball')
            ax.scatter(archive_behavior[:, 0], archive_behavior[:, 1], color='red', label='Archive')
            ax.scatter(pop_behavior[:, 0], pop_behavior[:, 1], color='blue', label='Population')
            ax.scatter(hof_behavior[:, 0], hof_behavior[:, 1], color='green', label='Hall of Fame')
            plt.legend()
            
        plt.show()
    
    else:
        possible_strats = ['random', 'least_novel', 'oldest', 'grid', 'grid_density', 'gmm', 'newest']
        colors = ['blue', 'red', 'yellow', 'green', 'pink', 'brown', 'purple']
        fig, ax = plt.subplots(3, 1, figsize=(20, 15))

        for s, archive_strat in enumerate(possible_strats):
            coverages = []
            uniformities = []
            rk_sim = []
            for i in range(N_EXP):
                print('experience', i, 'of strat', archive_strat)
                pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE,
                                                                     BD_BOUNDS,
                                                                     mini=True, archive_limit_size=ARCHIVE_LIMIT,
                                                                     archive_limit_strat=archive_strat,
                                                                     plot=PLOT, algo_type=ALGO, nb_gen=GEN,
                                                                     parallelize=PARALLELIZE,
                                                                     measures=True, pop_size=POP_SIZE,
                                                                     nb_cells=NB_CELLS, analyze_archive=True)
                cov = np.array(info[COV_TYPE])
                uni = np.array(info[UNI_TYPE])
                coverages.append(cov)
                uniformities.append(uni)
                ranking_similarities = np.array(info['ranking similarities'])
                rk_sim.append(ranking_similarities)
            mean_cov = np.mean(coverages, 0)
            std_cov = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
            mean_uni = np.mean(uniformities, 0)
            std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
            mean_rk_sim = np.mean(rk_sim, 0)
            std_rk = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
            ax[0].plot(mean_cov, label=archive_strat, lw=2, color=colors[s])
            ax[0].fill_between(list(range(GEN)), std_cov[0], std_cov[1], facecolor=colors[s], alpha=0.5)
            ax[1].plot(mean_uni, label=archive_strat, lw=2, color=colors[s])
            ax[1].fill_between(list(range(GEN)), std_uni[0], std_uni[1], facecolor=colors[s], alpha=0.5)
            ax[2].plot(mean_rk_sim, label=archive_strat, lw=2, color=colors[s])
            # ax[2].fill_between(list(range(len(mean_rk_sim))), std_rk[0], std_rk[1], facecolor=colors[s], alpha=0.5)
        
        coverages = []
        uniformities = []
        rk_sim = []
        for i in range(N_EXP):
            pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE,
                                                                 BD_BOUNDS,
                                                                 mini=True, archive_limit_size=None,
                                                                 plot=PLOT, algo_type=ALGO, nb_gen=GEN,
                                                                 parallelize=PARALLELIZE,
                                                                 measures=True, pop_size=POP_SIZE,
                                                                 nb_cells=NB_CELLS, analyze_archive=False)
            cov = np.array(info[COV_TYPE])
            uni = np.array(info[UNI_TYPE])
            coverages.append(cov)
            uniformities.append(uni)

        mean_cov = np.mean(coverages, 0)
        std_cov = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
        mean_uni = np.mean(uniformities, 0)
        std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]

        ax[0].plot(mean_cov, label='no archive limit', lw=2, color='gray')
        ax[0].fill_between(list(range(GEN)), std_cov[0], std_cov[1], facecolor='gray', alpha=0.5)
        ax[1].plot(mean_uni, label='no archive limit', lw=2, color='gray')
        ax[1].fill_between(list(range(GEN)), std_uni[0], std_uni[1], facecolor='gray', alpha=0.5)
        
        ax[0].set_xlabel("Generations", labelpad=15, fontsize=12, color="#333533")
        ax[1].set_xlabel("Generations", labelpad=15, fontsize=12, color="#333533")
        ax[2].set_xlabel("Iterations of reduction of archive", labelpad=15, fontsize=12, color="#333533")
        ax[0].set_ylabel("Mean " + COV_TYPE, labelpad=15, fontsize=12, color="#333533")
        ax[0].set_facecolor("#ffebb8")
        ax[0].legend(loc=4)
        ax[1].set_facecolor("#ffebb8")
        ax[1].set_ylabel("Mean " + UNI_TYPE, labelpad=15, fontsize=12, color="#333533")
        ax[1].legend(loc=2)
        ax[2].set_ylabel("Mean Kendall Tau similarity", labelpad=15, fontsize=12, color="#333533")
        ax[2].set_facecolor("#ffebb8")
        ax[2].legend(loc=4)

        plt.savefig('archive_analysis.png')
        plt.show()
