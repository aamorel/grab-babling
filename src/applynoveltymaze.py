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
GEN = 100
POP_SIZE = 100
ARCHIVE_LIMIT = 200
NB_CELLS = 100
BD_BOUNDS = [[0, 600], [0, 600]]
INITIAL_GENOTYPE_SIZE = 12
N_EXP = 30

ALGO = 'ns_rand'
PLOT = False
ARCHIVE_ANALYSIS = True
NOVELTY_ANALYSIS = True

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
        if not NOVELTY_ANALYSIS:
            possible_strats = ['random', 'least_novel', 'oldest', 'grid', 'grid_density', 'gmm', 'newest']
            colors = ['blue', 'red', 'yellow', 'green', 'pink', 'brown', 'purple']
            fig, ax = plt.subplots(3, 1, figsize=(20, 15))
            fig_2, ax_2 = plt.subplots(3, 1, figsize=(20, 15))

            for s, archive_strat in enumerate(possible_strats):
                coverages = []
                arch_coverages = []
                uniformities = []
                arch_uniformities = []
                rk_sim = []
                for i in range(N_EXP):
                    print('experience', i, 'of strat', archive_strat)
                    pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE,
                                                                         BD_BOUNDS,
                                                                         mini=False, archive_limit_size=ARCHIVE_LIMIT,
                                                                         archive_limit_strat=archive_strat,
                                                                         plot=PLOT, algo_type=ALGO, nb_gen=GEN,
                                                                         parallelize=PARALLELIZE, bound_genotype=1,
                                                                         measures=True, pop_size=POP_SIZE,
                                                                         nb_cells=NB_CELLS, analyze_archive=True)
                    cov = np.array(info['coverage'])
                    uni = np.array(info['uniformity'])
                    arch_cov = np.array(info['archive coverage'])
                    arch_uni = np.array(info['archive uniformity'])
                    coverages.append(cov)
                    uniformities.append(uni)
                    arch_uniformities.append(arch_uni)
                    arch_coverages.append(arch_cov)

                    ranking_similarities = np.array(info['ranking similarities'])
                    rk_sim.append(ranking_similarities)
                mean_cov = np.mean(coverages, 0)
                std_cov = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
                mean_uni = np.mean(uniformities, 0)
                std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
                mean_arch_cov = np.mean(arch_coverages, 0)
                std_arch_cov = [np.percentile(arch_coverages, 25, 0), np.percentile(arch_coverages, 75, 0)]
                mean_arch_uni = np.mean(arch_uniformities, 0)
                std_arch_uni = [np.percentile(arch_uniformities, 25, 0), np.percentile(arch_uniformities, 75, 0)]
                mean_rk_sim = np.mean(rk_sim, 0)
                std_rk = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
                ax[0].plot(mean_cov, label=archive_strat, lw=2, color=colors[s])
                ax[0].fill_between(list(range(GEN)), std_cov[0], std_cov[1], facecolor=colors[s], alpha=0.5)
                ax[1].plot(mean_uni, label=archive_strat, lw=2, color=colors[s])
                ax[1].fill_between(list(range(GEN)), std_uni[0], std_uni[1], facecolor=colors[s], alpha=0.5)
                ax[2].plot(mean_rk_sim, label=archive_strat, lw=2, color=colors[s])
                ax_2[0].plot(mean_arch_cov, label=archive_strat, lw=2, color=colors[s])
                ax_2[0].fill_between(list(range(GEN)), std_arch_cov[0], std_arch_cov[1], facecolor=colors[s], alpha=0.5)
                ax_2[1].plot(mean_arch_uni, label=archive_strat, lw=2, color=colors[s])
                ax_2[1].fill_between(list(range(GEN)), std_arch_uni[0], std_arch_uni[1], facecolor=colors[s], alpha=0.5)
                ax_2[2].plot(mean_rk_sim, label=archive_strat, lw=2, color=colors[s])
                # ax[2].fill_between(list(range(len(mean_rk_sim))), std_rk[0], std_rk[1],
                #                    facecolor=colors[s], alpha=0.5)
            
            coverages = []
            arch_coverages = []
            uniformities = []
            arch_uniformities = []
            rk_sim = []
            for i in range(N_EXP):
                pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE,
                                                                     BD_BOUNDS,
                                                                     mini=False, archive_limit_size=None,
                                                                     plot=PLOT, algo_type=ALGO, nb_gen=GEN,
                                                                     parallelize=PARALLELIZE, bound_genotype=1,
                                                                     measures=True, pop_size=POP_SIZE,
                                                                     nb_cells=NB_CELLS, analyze_archive=False)
                cov = np.array(info['coverage'])
                uni = np.array(info['uniformity'])
                coverages.append(cov)
                uniformities.append(uni)
                arch_cov = np.array(info['archive coverage'])
                arch_uni = np.array(info['archive uniformity'])
                arch_coverages.append(arch_cov)
                arch_uniformities.append(arch_uni)

            mean_cov = np.mean(coverages, 0)
            std_cov = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
            mean_uni = np.mean(uniformities, 0)
            std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
            mean_arch_cov = np.mean(arch_coverages, 0)
            std_arch_cov = [np.percentile(arch_coverages, 25, 0), np.percentile(arch_coverages, 75, 0)]
            mean_arch_uni = np.mean(arch_uniformities, 0)
            std_arch_uni = [np.percentile(arch_uniformities, 25, 0), np.percentile(arch_uniformities, 75, 0)]

            ax[0].plot(mean_cov, label='no archive limit', lw=2, color='gray')
            ax[0].fill_between(list(range(GEN)), std_cov[0], std_cov[1], facecolor='gray', alpha=0.5)
            ax[1].plot(mean_uni, label='no archive limit', lw=2, color='gray')
            ax[1].fill_between(list(range(GEN)), std_uni[0], std_uni[1], facecolor='gray', alpha=0.5)
            ax_2[0].plot(mean_arch_cov, label='no archive limit', lw=2, color='gray')
            ax_2[0].fill_between(list(range(GEN)), std_arch_cov[0], std_arch_cov[1], facecolor='gray', alpha=0.5)
            ax_2[1].plot(mean_arch_uni, label='no archive limit', lw=2, color='gray')
            ax_2[1].fill_between(list(range(GEN)), std_arch_uni[0], std_arch_uni[1], facecolor='gray', alpha=0.5)
            
            ax[0].set_xlabel("Generations", labelpad=15, fontsize=12, color="#333533")
            ax[1].set_xlabel("Generations", labelpad=15, fontsize=12, color="#333533")
            ax[2].set_xlabel("Iterations of reduction of archive", labelpad=15, fontsize=12, color="#333533")
            ax[0].set_ylabel("Mean coverage", labelpad=15, fontsize=12, color="#333533")
            ax[0].set_facecolor("#ffebb8")
            ax[0].legend(loc=4)
            ax[1].set_facecolor("#ffebb8")
            ax[1].set_ylabel("Mean uniformity", labelpad=15, fontsize=12, color="#333533")
            ax[1].legend(loc=2)
            ax[2].set_ylabel("Mean Kendall Tau similarity", labelpad=15, fontsize=12, color="#333533")
            ax[2].set_facecolor("#ffebb8")
            ax[2].legend(loc=4)

            ax_2[0].set_xlabel("Generations", labelpad=15, fontsize=12, color="#333533")
            ax_2[1].set_xlabel("Generations", labelpad=15, fontsize=12, color="#333533")
            ax_2[2].set_xlabel("Iterations of reduction of archive", labelpad=15, fontsize=12, color="#333533")
            ax_2[0].set_ylabel("Mean archive coverage", labelpad=15, fontsize=12, color="#333533")
            ax_2[0].set_facecolor("#ffebb8")
            ax_2[0].legend(loc=4)
            ax_2[1].set_facecolor("#ffebb8")
            ax_2[1].set_ylabel("Mean archive uniformity", labelpad=15, fontsize=12, color="#333533")
            ax_2[1].legend(loc=2)
            ax_2[2].set_ylabel("Mean Kendall Tau similarity", labelpad=15, fontsize=12, color="#333533")
            ax_2[2].set_facecolor("#ffebb8")
            ax_2[2].legend(loc=4)

            fig.savefig('full_analysis_maze.png')
            fig_2.savefig('archive_analysis_maze.png')
            if PLOT:
                plt.show()
        else:
            possible_degrees = [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
            colors = ['blue', 'red', 'yellow', 'green', 'pink', 'brown', 'purple']
            fig, ax = plt.subplots(3, 1, figsize=(20, 15))
            fig_2, ax_2 = plt.subplots(3, 1, figsize=(20, 15))

            for s, archive_strat in enumerate(possible_strats):
                coverages = []
                arch_coverages = []
                uniformities = []
                arch_uniformities = []
                rk_sim = []
                for i in range(N_EXP):
                    print('experience', i, 'of strat', archive_strat)
                    pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE,
                                                                         BD_BOUNDS, altered_novelty=True,
                                                                         alteration_degree=archive_strat,
                                                                         mini=False,
                                                                         plot=PLOT, algo_type=ALGO, nb_gen=GEN,
                                                                         parallelize=PARALLELIZE, bound_genotype=1,
                                                                         measures=True, pop_size=POP_SIZE,
                                                                         nb_cells=NB_CELLS)
                    cov = np.array(info['coverage'])
                    uni = np.array(info['uniformity'])
                    arch_cov = np.array(info['archive coverage'])
                    arch_uni = np.array(info['archive uniformity'])
                    coverages.append(cov)
                    uniformities.append(uni)
                    arch_uniformities.append(arch_uni)
                    arch_coverages.append(arch_cov)

                    ranking_similarities = np.array(info['ranking similarities novelty'])
                    rk_sim.append(ranking_similarities)
                mean_cov = np.mean(coverages, 0)
                std_cov = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
                mean_uni = np.mean(uniformities, 0)
                std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
                mean_arch_cov = np.mean(arch_coverages, 0)
                std_arch_cov = [np.percentile(arch_coverages, 25, 0), np.percentile(arch_coverages, 75, 0)]
                mean_arch_uni = np.mean(arch_uniformities, 0)
                std_arch_uni = [np.percentile(arch_uniformities, 25, 0), np.percentile(arch_uniformities, 75, 0)]
                mean_rk_sim = np.mean(rk_sim, 0)
                std_rk = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
                ax[0].plot(mean_cov, label=archive_strat, lw=2, color=colors[s])
                ax[0].fill_between(list(range(GEN)), std_cov[0], std_cov[1], facecolor=colors[s], alpha=0.5)
                ax[1].plot(mean_uni, label=archive_strat, lw=2, color=colors[s])
                ax[1].fill_between(list(range(GEN)), std_uni[0], std_uni[1], facecolor=colors[s], alpha=0.5)
                ax[2].plot(mean_rk_sim, label=archive_strat, lw=2, color=colors[s])
                ax_2[0].plot(mean_arch_cov, label=archive_strat, lw=2, color=colors[s])
                ax_2[0].fill_between(list(range(GEN)), std_arch_cov[0], std_arch_cov[1], facecolor=colors[s], alpha=0.5)
                ax_2[1].plot(mean_arch_uni, label=archive_strat, lw=2, color=colors[s])
                ax_2[1].fill_between(list(range(GEN)), std_arch_uni[0], std_arch_uni[1], facecolor=colors[s], alpha=0.5)
                ax_2[2].plot(mean_rk_sim, label=archive_strat, lw=2, color=colors[s])
                # ax[2].fill_between(list(range(len(mean_rk_sim))), std_rk[0], std_rk[1],
                #                    facecolor=colors[s], alpha=0.5)
            
            coverages = []
            arch_coverages = []
            uniformities = []
            arch_uniformities = []
            rk_sim = []
            for i in range(N_EXP):
                pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE,
                                                                     BD_BOUNDS, altered_novelty=False,
                                                                     mini=False,
                                                                     plot=PLOT, algo_type=ALGO, nb_gen=GEN,
                                                                     parallelize=PARALLELIZE, bound_genotype=1,
                                                                     measures=True, pop_size=POP_SIZE,
                                                                     nb_cells=NB_CELLS)
                cov = np.array(info['coverage'])
                uni = np.array(info['uniformity'])
                coverages.append(cov)
                uniformities.append(uni)
                arch_cov = np.array(info['archive coverage'])
                arch_uni = np.array(info['archive uniformity'])
                arch_coverages.append(arch_cov)
                arch_uniformities.append(arch_uni)

            mean_cov = np.mean(coverages, 0)
            std_cov = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
            mean_uni = np.mean(uniformities, 0)
            std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
            mean_arch_cov = np.mean(arch_coverages, 0)
            std_arch_cov = [np.percentile(arch_coverages, 25, 0), np.percentile(arch_coverages, 75, 0)]
            mean_arch_uni = np.mean(arch_uniformities, 0)
            std_arch_uni = [np.percentile(arch_uniformities, 25, 0), np.percentile(arch_uniformities, 75, 0)]

            ax[0].plot(mean_cov, label='no alteration', lw=2, color='gray')
            ax[0].fill_between(list(range(GEN)), std_cov[0], std_cov[1], facecolor='gray', alpha=0.5)
            ax[1].plot(mean_uni, label='no alteration', lw=2, color='gray')
            ax[1].fill_between(list(range(GEN)), std_uni[0], std_uni[1], facecolor='gray', alpha=0.5)
            ax_2[0].plot(mean_arch_cov, label='no alteration', lw=2, color='gray')
            ax_2[0].fill_between(list(range(GEN)), std_arch_cov[0], std_arch_cov[1], facecolor='gray', alpha=0.5)
            ax_2[1].plot(mean_arch_uni, label='no alteration', lw=2, color='gray')
            ax_2[1].fill_between(list(range(GEN)), std_arch_uni[0], std_arch_uni[1], facecolor='gray', alpha=0.5)
            
            ax[0].set_xlabel("Generations", labelpad=15, fontsize=12, color="#333533")
            ax[1].set_xlabel("Generations", labelpad=15, fontsize=12, color="#333533")
            ax[2].set_xlabel("Iterations of novelty computation", labelpad=15, fontsize=12, color="#333533")
            ax[0].set_ylabel("Mean coverage", labelpad=15, fontsize=12, color="#333533")
            ax[0].set_facecolor("#ffebb8")
            ax[0].legend(loc=4)
            ax[1].set_facecolor("#ffebb8")
            ax[1].set_ylabel("Mean uniformity", labelpad=15, fontsize=12, color="#333533")
            ax[1].legend(loc=2)
            ax[2].set_ylabel("Mean Kendall Tau similarity", labelpad=15, fontsize=12, color="#333533")
            ax[2].set_facecolor("#ffebb8")
            ax[2].legend(loc=4)

            ax_2[0].set_xlabel("Generations", labelpad=15, fontsize=12, color="#333533")
            ax_2[1].set_xlabel("Generations", labelpad=15, fontsize=12, color="#333533")
            ax_2[2].set_xlabel("Iterations of novelty computation", labelpad=15, fontsize=12, color="#333533")
            ax_2[0].set_ylabel("Mean archive coverage", labelpad=15, fontsize=12, color="#333533")
            ax_2[0].set_facecolor("#ffebb8")
            ax_2[0].legend(loc=4)
            ax_2[1].set_facecolor("#ffebb8")
            ax_2[1].set_ylabel("Mean archive uniformity", labelpad=15, fontsize=12, color="#333533")
            ax_2[1].legend(loc=2)
            ax_2[2].set_ylabel("Mean Kendall Tau similarity", labelpad=15, fontsize=12, color="#333533")
            ax_2[2].set_facecolor("#ffebb8")
            ax_2[2].legend(loc=4)

            fig.savefig('full_analysis_maze_novelty.png')
            fig_2.savefig('archive_analysis_maze_novelty.png')
            if PLOT:
                plt.show()
