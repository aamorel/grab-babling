import gym
import gym_fastsim  # must still be imported
import noveltysearch
import time
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator

DISPLAY = False
PARALLELIZE = True
GEN = 200
POP_SIZE = 10
ARCHIVE_LIMIT = 200
NB_CELLS = 100
BD_BOUNDS = [[0, 600], [0, 600]]
INITIAL_GENOTYPE_SIZE = 12
N_EXP = 30
MINI = True

ALGO = 'ns_rand'
PLOT = False
ARCHIVE_ANALYSIS = False
NOVELTY_ANALYSIS = False
SIMPLE_RUN = False

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
    if MINI:
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

        if(DISPLAY):
            time.sleep(0.01)
        if eo:
            break
    # use last info to compute behavior and fitness
    behavior = [info["robot_pos"][0], info["robot_pos"][1]]
    fitness = info["dist_obj"]
    return (behavior, (fitness,), info)


if __name__ == "__main__":
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=BIGGER_SIZE, weight='bold')          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE, titleweight='bold')     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE, labelweight='bold')    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE, titleweight='bold')  # fontsize of the figure title

    if not ARCHIVE_ANALYSIS:
        if SIMPLE_RUN:

            archive_strat = 'least_novel'
            
            pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE, BD_BOUNDS,
                                                                 mini=True, archive_limit_size=None,
                                                                 archive_limit_strat=archive_strat,
                                                                 plot=PLOT, algo_type='ns_rand', nb_gen=GEN,
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
            # ############################### ANALYSI IMPORTANCE OF ARCHIVE #####################################
            fig, ax = plt.subplots(2, 1, figsize=(20, 15))
            # adding a run for classic ns
            coverages = []
            uniformities = []
            for i in range(N_EXP):
                pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE,
                                                                     BD_BOUNDS,
                                                                     mini=True, archive_limit_size=None,
                                                                     plot=PLOT, algo_type='ns_rand', nb_gen=GEN,
                                                                     parallelize=PARALLELIZE, bound_genotype=1,
                                                                     measures=True, pop_size=POP_SIZE,
                                                                     nb_cells=NB_CELLS)
                cov = np.array(info['coverage'])
                uni = np.array(info['uniformity'])
                coverages.append(cov)
                uniformities.append(uni)

            mean_cov = np.mean(coverages, 0)
            std_cov = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
            sig_cov = [np.std(coverages, 0) - mean_cov, np.std(coverages, 0) + mean_cov]
            mean_uni = np.mean(uniformities, 0)
            std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
            sig_uni = [np.std(uniformities, 0) - mean_uni, np.std(uniformities, 0) + mean_uni]

            ax[0].plot(mean_cov, label='classic ns', lw=2, color='grey')
            ax[0].fill_between(list(range(GEN)), sig_cov[0], sig_cov[1], facecolor='grey', alpha=0.5)
            ax[1].plot(mean_uni, label='classic ns', lw=2, color='grey')
            ax[1].fill_between(list(range(GEN)), sig_uni[0], sig_uni[1], facecolor='grey', alpha=0.5)

            # adding a run for no archive ns
            coverages = []
            uniformities = []
            for i in range(N_EXP):
                pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE,
                                                                     BD_BOUNDS,
                                                                     mini=True, archive_limit_size=None, nb_gen=GEN,
                                                                     plot=PLOT, algo_type='ns_no_archive',
                                                                     parallelize=PARALLELIZE, bound_genotype=1,
                                                                     measures=True, pop_size=POP_SIZE,
                                                                     nb_cells=NB_CELLS, analyze_archive=False)
                cov = np.array(info['coverage'])
                uni = np.array(info['uniformity'])
                coverages.append(cov)
                uniformities.append(uni)

            mean_cov = np.mean(coverages, 0)
            std_cov = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
            sig_cov = [np.std(coverages, 0) - mean_cov, np.std(coverages, 0) + mean_cov]
            mean_uni = np.mean(uniformities, 0)
            std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
            sig_uni = [np.std(uniformities, 0) - mean_uni, np.std(uniformities, 0) + mean_uni]

            ax[0].plot(mean_cov, label='no archive', lw=2, color='green')
            ax[0].fill_between(list(range(GEN)), sig_cov[0], sig_cov[1], facecolor='green', alpha=0.5)
            ax[1].plot(mean_uni, label='no archive', lw=2, color='green')
            ax[1].fill_between(list(range(GEN)), sig_uni[0], sig_uni[1], facecolor='green', alpha=0.5)

            # adding a run for random search
            coverages = []
            uniformities = []
            for i in range(N_EXP):
                pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE,
                                                                     BD_BOUNDS,
                                                                     mini=True, archive_limit_size=None, nb_gen=GEN,
                                                                     plot=PLOT, algo_type='random_search',
                                                                     parallelize=PARALLELIZE, bound_genotype=1,
                                                                     measures=True, pop_size=POP_SIZE,
                                                                     nb_cells=NB_CELLS, analyze_archive=False)
                cov = np.array(info['coverage'])
                uni = np.array(info['uniformity'])
                coverages.append(cov)
                uniformities.append(uni)

            mean_cov = np.mean(coverages, 0)
            std_cov = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
            sig_cov = [np.std(coverages, 0) - mean_cov, np.std(coverages, 0) + mean_cov]
            mean_uni = np.mean(uniformities, 0)
            std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
            sig_uni = [np.std(uniformities, 0) - mean_uni, np.std(uniformities, 0) + mean_uni]

            ax[0].plot(mean_cov, label='random search', lw=2, color='orange')
            ax[0].fill_between(list(range(GEN)), sig_cov[0], sig_cov[1], facecolor='orange', alpha=0.5)
            ax[1].plot(mean_uni, label='random search', lw=2, color='orange')
            ax[1].fill_between(list(range(GEN)), sig_uni[0], sig_uni[1], facecolor='orange', alpha=0.5)

            # generating the plot
            ax[0].set_xlabel("Generations", labelpad=15, fontsize=12, color="#333533")
            ax[1].set_xlabel("Generations", labelpad=15, fontsize=12, color="#333533")
            ax[0].set_ylabel("Mean coverage", labelpad=15, fontsize=12, color="#333533")
            ax[0].set_facecolor("#ffebb8")
            ax[0].legend(loc=4)
            ax[1].set_facecolor("#ffebb8")
            ax[1].set_ylabel("Mean uniformity", labelpad=15, fontsize=12, color="#333533")
            ax[1].legend(loc=2)

            fig.savefig('archive_importance_maze.png')
            if PLOT:
                plt.show()
    else:
        if not NOVELTY_ANALYSIS:
            # ################################## ARCHIVE MANAGEMENT ANALYSIS ####################################
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
                                                                         mini=True, archive_limit_size=ARCHIVE_LIMIT,
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
            
            # adding a run for classic ns
            coverages = []
            arch_coverages = []
            uniformities = []
            arch_uniformities = []
            rk_sim = []
            for i in range(N_EXP):
                pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE,
                                                                     BD_BOUNDS,
                                                                     mini=True, archive_limit_size=None,
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

            # adding a run for random search
            coverages = []
            arch_coverages = []
            uniformities = []
            arch_uniformities = []
            rk_sim = []
            for i in range(N_EXP):
                pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE,
                                                                     BD_BOUNDS,
                                                                     mini=True, archive_limit_size=None,
                                                                     plot=PLOT, algo_type='random_search', nb_gen=GEN,
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

            ax[0].plot(mean_cov, label='random search', lw=2, color='orange')
            ax[0].fill_between(list(range(GEN)), std_cov[0], std_cov[1], facecolor='orange', alpha=0.5)
            ax[1].plot(mean_uni, label='random search', lw=2, color='orange')
            ax[1].fill_between(list(range(GEN)), std_uni[0], std_uni[1], facecolor='orange', alpha=0.5)
            ax_2[0].plot(mean_arch_cov, label='random search', lw=2, color='orange')
            ax_2[0].fill_between(list(range(GEN)), std_arch_cov[0], std_arch_cov[1], facecolor='orange', alpha=0.5)
            ax_2[1].plot(mean_arch_uni, label='random search', lw=2, color='orange')
            ax_2[1].fill_between(list(range(GEN)), std_arch_uni[0], std_arch_uni[1], facecolor='orange', alpha=0.5)

            # adding a run for fitness ea
            coverages = []
            arch_coverages = []
            uniformities = []
            arch_uniformities = []
            rk_sim = []
            for i in range(N_EXP):
                pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE,
                                                                     BD_BOUNDS,
                                                                     mini=True, archive_limit_size=None,
                                                                     plot=PLOT, algo_type='classic_ea', nb_gen=GEN,
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

            ax[0].plot(mean_cov, label='fitness ea', lw=2, color='cyan')
            ax[0].fill_between(list(range(GEN)), std_cov[0], std_cov[1], facecolor='cyan', alpha=0.5)
            ax[1].plot(mean_uni, label='fitness ea', lw=2, color='cyan')
            ax[1].fill_between(list(range(GEN)), std_uni[0], std_uni[1], facecolor='cyan', alpha=0.5)
            ax_2[0].plot(mean_arch_cov, label='fitness ea', lw=2, color='cyan')
            ax_2[0].fill_between(list(range(GEN)), std_arch_cov[0], std_arch_cov[1], facecolor='cyan', alpha=0.5)
            ax_2[1].plot(mean_arch_uni, label='fitness ea', lw=2, color='cyan')
            ax_2[1].fill_between(list(range(GEN)), std_arch_uni[0], std_arch_uni[1], facecolor='cyan', alpha=0.5)

            # generating the plot
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
            # ################################# ANALYSIS OF ALTERATION OF NOVELTY ######################################

            # looping through all degrees
            possible_degrees = [0.1, 0.5, 1, 5, 10, 20, 100]
            colors = ['blue', 'red', 'yellow', 'green', 'pink', 'brown', 'purple']
            fig, ax = plt.subplots(3, 1, figsize=(20, 15))
            fig_2, ax_2 = plt.subplots(3, 1, figsize=(20, 15))

            for s, archive_strat in enumerate(possible_degrees):
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
                                                                         mini=True,
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
            
            # adding a run for classic ns
            coverages = []
            arch_coverages = []
            uniformities = []
            arch_uniformities = []
            rk_sim = []
            for i in range(N_EXP):
                pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE,
                                                                     BD_BOUNDS, altered_novelty=False,
                                                                     mini=True,
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

            # adding a run for random search
            coverages = []
            arch_coverages = []
            uniformities = []
            arch_uniformities = []
            rk_sim = []
            for i in range(N_EXP):
                pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE,
                                                                     BD_BOUNDS, altered_novelty=False,
                                                                     mini=True,
                                                                     plot=PLOT, algo_type='random_search', nb_gen=GEN,
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

            ax[0].plot(mean_cov, label='random search', lw=2, color='orange')
            ax[0].fill_between(list(range(GEN)), std_cov[0], std_cov[1], facecolor='orange', alpha=0.5)
            ax[1].plot(mean_uni, label='random search', lw=2, color='orange')
            ax[1].fill_between(list(range(GEN)), std_uni[0], std_uni[1], facecolor='orange', alpha=0.5)
            ax_2[0].plot(mean_arch_cov, label='random search', lw=2, color='orange')
            ax_2[0].fill_between(list(range(GEN)), std_arch_cov[0], std_arch_cov[1], facecolor='orange', alpha=0.5)
            ax_2[1].plot(mean_arch_uni, label='random search', lw=2, color='orange')
            ax_2[1].fill_between(list(range(GEN)), std_arch_uni[0], std_arch_uni[1], facecolor='orange', alpha=0.5)

            # adding a run for fitness ea
            coverages = []
            arch_coverages = []
            uniformities = []
            arch_uniformities = []
            rk_sim = []
            for i in range(N_EXP):
                pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE,
                                                                     BD_BOUNDS, altered_novelty=False,
                                                                     mini=True,
                                                                     plot=PLOT, algo_type='classic_ea', nb_gen=GEN,
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

            ax[0].plot(mean_cov, label='fitness ea', lw=2, color='cyan')
            ax[0].fill_between(list(range(GEN)), std_cov[0], std_cov[1], facecolor='cyan', alpha=0.5)
            ax[1].plot(mean_uni, label='fitness ea', lw=2, color='cyan')
            ax[1].fill_between(list(range(GEN)), std_uni[0], std_uni[1], facecolor='cyan', alpha=0.5)
            ax_2[0].plot(mean_arch_cov, label='fitness ea', lw=2, color='cyan')
            ax_2[0].fill_between(list(range(GEN)), std_arch_cov[0], std_arch_cov[1], facecolor='cyan', alpha=0.5)
            ax_2[1].plot(mean_arch_uni, label='fitness ea', lw=2, color='cyan')
            ax_2[1].fill_between(list(range(GEN)), std_arch_uni[0], std_arch_uni[1], facecolor='cyan', alpha=0.5)
            
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

            # generating the plot
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
