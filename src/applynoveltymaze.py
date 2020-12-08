import gym
import gym_fastsim  # must still be imported
import noveltysearch
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator

EXAMPLE = False
DISPLAY = False
PARALLELIZE = True
GEN = 50
POP_SIZE = 100
ARCHIVE_LIMIT = 200
NB_CELLS = 100
BD_BOUNDS = [[0, 600], [0, 600]]
INITIAL_GENOTYPE_SIZE = 12
N_EXP = 30

ALGO = 'ns_rand'
PLOT = False
ARCHIVE_ANALYSIS = True

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


def evaluate_individual(individual):
    """Evaluates an individual: computes its value in the behavior descriptor space,
    and its fitness value.

    Args:
        individual (Individual): an individual

    Returns:
        tuple: tuple of behavior (list) and fitness(tuple)
    """
    info = {}
    if EXAMPLE:
        # example: behavior is juste genotype
        behavior = individual.copy()
        # example: Rastrigin function
        dim = len(individual)
        A = 10
        fitness = 0
        for i in range(dim):
            fitness += individual[i]**2 - A * math.cos(2 * math.pi * individual[i])
        fitness += A * dim
        return (behavior, (fitness,), info)
    else:
        env = gym.make('FastsimSimpleNavigation-v0')
        env.reset()
        if(DISPLAY):
            env.enable_display()

        action = [0, 0]

        for i in range(1500):
            env.render()
            # apply previously chosen action
            o, r, eo, info = env.step(action)
            # normalize observations from sensor
            o[0] = o[0] / env.maxSensorRange
            o[1] = o[1] / env.maxSensorRange
            o[2] = o[2] / env.maxSensorRange

            # CONTROLLER
            # 5 inputs, 2 outputs
            individual = np.array(individual)
            o = np.array(o)
            action[0] = np.sum(np.multiply(individual[0:5], o)) + individual[5]
            action[1] = np.sum(np.multiply(individual[6:11], o)) + individual[11]

            # print("Step %d Obs=%s  reward=%f  dist. to objective=%f  robot position=%s  End of ep=%s" % (i, str(o), r,
            #     info["dist_obj"], str(info["robot_pos"]), str(eo)))
            if(DISPLAY):
                time.sleep(0.01)
            if eo:
                break
        # use last info to compute behavior and fitness
        behavior = [info["robot_pos"][0], info["robot_pos"][1]]
        fitness = info["dist_obj"]
        return (behavior, (fitness,), info)


if __name__ == "__main__":
    if not ARCHIVE_ANALYSIS:

        archive_strat = 'least_novel'
        
        pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE, BD_BOUNDS,
                                                             mini=True, archive_limit_size=ARCHIVE_LIMIT,
                                                             archive_limit_strat=archive_strat,
                                                             plot=PLOT, algo_type=ALGO, nb_gen=GEN,
                                                             parallelize=PARALLELIZE,
                                                             measures=True, pop_size=POP_SIZE, nb_cells=NB_CELLS)

        if PLOT:
            # plot final states
            env = gym.make('FastsimSimpleNavigation-v0')
            env.reset()
            maze = env.map.get_data()
            maze = np.array([str(pix) for pix in maze])
            maze[maze == 'status_t.obstacle'] = 0.0
            maze[maze == 'status_t.free'] = 1.0
            maze = np.reshape(maze, (200, 200))
            maze = np.array(maze, dtype='float')
            archive_behavior = np.array([ind.behavior_descriptor.values for ind in archive])
            pop_behavior = np.array([ind.behavior_descriptor.values for ind in pop])
            hof_behavior = np.array([ind.behavior_descriptor.values for ind in hof])
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set(title='Final Archive', xlabel='x1', ylabel='x2')
            ax.imshow(maze)
            ax.scatter(archive_behavior[:, 0] / 3, archive_behavior[:, 1] / 3, color='red', label='Archive')
            ax.scatter(pop_behavior[:, 0] / 3, pop_behavior[:, 1] / 3, color='blue', label='Population')
            ax.scatter(hof_behavior[:, 0] / 3, hof_behavior[:, 1] / 3, color='green', label='Hall of Fame')
            plt.legend()
            
        plt.show()
    
    else:
        possible_strats = ['random', 'least_novel', 'oldest', 'grid', 'grid_density', 'gmm']
        colors = ['blue', 'red', 'yellow', 'green', 'pink', 'brown']
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(20, 10))

        for s, archive_strat in enumerate(possible_strats):
            coverages = []
            uniformities = []
            for i in range(N_EXP):
                print('experience', i, 'of strat', archive_strat)
                pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE,
                                                                     BD_BOUNDS,
                                                                     mini=True, archive_limit_size=ARCHIVE_LIMIT,
                                                                     archive_limit_strat=archive_strat,
                                                                     plot=PLOT, algo_type=ALGO, nb_gen=GEN,
                                                                     parallelize=PARALLELIZE,
                                                                     measures=True, pop_size=POP_SIZE,
                                                                     nb_cells=NB_CELLS)
                cov = np.array(info['archive coverage'])
                uni = np.array(info['archive uniformity'])
                coverages.append(cov)
                uniformities.append(uni)
            mean_cov = np.mean(coverages, 0)
            std_cov = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
            mean_uni = np.mean(uniformities, 0)
            std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
            ax[0].plot(mean_cov, label=archive_strat, lw=2, color=colors[s])
            ax[0].fill_between(list(range(GEN)), std_cov[0], std_cov[1], facecolor=colors[s], alpha=0.5)
            ax[1].plot(mean_uni, label=archive_strat, lw=2, color=colors[s])
            ax[1].fill_between(list(range(GEN)), std_uni[0], std_uni[1], facecolor=colors[s], alpha=0.5)

        ax[1].set_xlabel("Generations", labelpad=15, fontsize=12, color="#333533")
        ax[0].set_ylabel("Mean coverage", labelpad=15, fontsize=12, color="#333533")
        ax[0].set_facecolor("#ffebb8")
        ax[0].legend(loc=4)
        ax[1].set_facecolor("#ffebb8")
        ax[1].set_ylabel("Mean uniformity", labelpad=15, fontsize=12, color="#333533")
        ax[1].legend(loc=2)
        plt.savefig('archive_analysis.png')
        plt.show()
