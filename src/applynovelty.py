import gym
import noveltysearch
import time
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator
import utils
import math

import gym_fastsim  # must still be imported
import slimevolleygym  # must still be imported
import ballbeam_gym  # must still be imported
import gym_minigrid  # must still be imported
from gym_minigrid.wrappers import ImgObsWrapper  # must still be imported


DISPLAY = True
PARALLELIZE = False
GEN = 2
POP_SIZE = 10
ARCHIVE_LIMIT = 20
NB_CELLS = 100
N_EXP = 60
ALGO = 'ns_rand'
PLOT = False
ARCHIVE_ANALYSIS = False
NOVELTY_ANALYSIS = False
SIMPLE_RUN = True
ENV_NAME = 'slime'
SHOW_HOF = False

if ENV_NAME == 'maze':
    BD_BOUNDS = [[0, 600], [0, 600]]
    INITIAL_GENOTYPE_SIZE = 12
    MINI = True

if ENV_NAME == 'slime':
    # global variable for the environment
    ENV = gym.make("SlimeVolley-v0")
    BD_BOUNDS = [[0, 1], [0, 1]]
    INITIAL_GENOTYPE_SIZE = 99
    # depends on the version of slimevolley (random or determinist)
    N_REPEAT = 1
    MINI = False

if ENV_NAME == 'beam':
    # pass env arguments as kwargs
    kwargs = {'timestep': 0.05,
              'beam_length': 1.0,
              'max_angle': 0.4,
              'init_velocity': 0.0,
              'max_timesteps': 500,
              'action_mode': 'discrete'}

    # create env
    ENV = gym.make('BallBeamThrow-v0', **kwargs)
    BD_BOUNDS = [[0, 3]]
    INITIAL_GENOTYPE_SIZE = 11
    MINI = False

if ENV_NAME == 'grid':
    # create env
    ENV = ImgObsWrapper(gym.make('MiniGrid-Empty-8x8-v0'))
    BD_BOUNDS = [[0, 7], [0, 7]]
    NB_CELLS = 64
    INITIAL_GENOTYPE_SIZE = 11
    MINI = False

if ENV_NAME == 'bipedal':
    # global variable for the environment
    ENV = gym.make('BipedalWalker-v3')
    BD_BOUNDS = [[-1, 1], [0, 1]]
    INITIAL_GENOTYPE_SIZE = 118
    MINI = False

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


def evaluate_slime(individual):
    """Evaluates an individual: computes its value in the behavior descriptor space,
    and its fitness value.

    Args:
        individual (Individual): an individual

    Returns:
        tuple: tuple of behavior (list) and fitness(tuple)
    """
    global ENV
    inf = {}
    fitness_arr = []
    behavior_arr = []

    for _ in range(N_REPEAT):
        ENV.reset()

        action = [0, 0, 0]
        eo = False
        count = 0
        reward = 0
        distance_ball = 0
        distance_player = 0

        # CONTROLLER
        ind = np.array(individual)
        controller = utils.NeuralAgentNumpy(12, 3, n_hidden_layers=1, n_neurons_per_hidden=6)
        controller.set_weights(ind)
        while not eo:
            count += 1
            if DISPLAY:
                ENV.render()
            # apply previously chosen action
            o, r, eo, info = ENV.step(action)
            reward += r

            action = controller.choose_action(o)
            action = [int(a > 0) for a in action]

            dist_to_ball = math.sqrt((o[0] - o[4])**2 + (o[1] - o[5])**2)
            distance_ball += dist_to_ball

            dist_to_player = math.sqrt((o[0] - o[8])**2 + (o[1] - o[9])**2)
            distance_player += dist_to_player

            if(DISPLAY):
                time.sleep(0.01)

        # use last info to compute behavior and fitness
        mean_distance_ball = distance_ball / (count * 4)
        mean_distance_player = distance_player / (count * 4)
        
        # variant 1: game duration + distance to ball
        # behavior_arr.append([count / 3000, mean_distance_ball])

        # variant 2: distance to other player + distance to ball
        behavior_arr.append([mean_distance_player, mean_distance_ball])

        fitness_arr.append(reward)

    behavior_arr = np.array(behavior_arr)
    behavior = np.mean(behavior_arr, axis=0)
    fitness_arr = np.array(fitness_arr)
    fitness = np.mean(fitness_arr)
    return (behavior, (fitness,), inf)


def evaluate_maze(individual):
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


def evaluate_beam(individual):
    """Evaluates an individual: computes its value in the behavior descriptor space,
    and its fitness value.

    Args:
        individual (Individual): an individual

    Returns:
        tuple: tuple of behavior (list) and fitness(tuple)
    """
    info = {}

    global ENV
    ENV.reset()

    action = 1
    rew = 0

    while True:
        if DISPLAY:
            ENV.render()
        # apply previously chosen action
        o, r, eo, info = ENV.step(action)

        # CONTROLLER
        # 3 inputs, 1 output
        individual = np.array(individual)
        o = np.array(o)
        hid_0 = np.sum(np.multiply(individual[0:3], o)) + individual[3]
        hid_1 = np.sum(np.multiply(individual[4:7], o)) + individual[7]
        hid = np.array([hid_0, hid_1])
        hid = 1 / (1 + np.exp(-hid))

        res = np.sum(np.multiply(individual[8:10], hid) + individual[10])
        res = 1 / (1 + np.exp(-res))
        # res = random.random()
        if res < 0.33:
            action = 0
        elif res < 0.66:
            action = 1
        else:
            action = 2

        rew += r
        if eo:
            break

    behavior = [r]
    fitness = r

    return (behavior, (fitness,), info)


def evaluate_grid(individual):
    """Evaluates an individual: computes its value in the behavior descriptor space,
    and its fitness value.

    Args:
        individual (Individual): an individual

    Returns:
        tuple: tuple of behavior (list) and fitness(tuple)
    """
    info = {}

    global ENV
    ENV.reset()

    action = 0
    rew = 0

    for i in range(100):
        if DISPLAY:
            ENV.render()
        # apply previously chosen action
        o, r, eo, info = ENV.step(action)

        # CONTROLLER
        # 2 inputs, 1 output
        individual = np.array(individual)
        o = list(ENV.agent_pos)
        o.append(ENV.agent_dir)
        o = np.array(o)
        hid_0 = np.sum(np.multiply(individual[0:3], o)) + individual[3]
        hid_1 = np.sum(np.multiply(individual[4:7], o)) + individual[7]
        hid = np.array([hid_0, hid_1])
        hid = 1 / (1 + np.exp(-hid))

        res = np.sum(np.multiply(individual[8:10], hid) + individual[10])
        res = 1 / (1 + np.exp(-res))
        # res = random.random()
        if res < 0.33:
            action = 0
        elif res < 0.66:
            action = 1
        else:
            action = 2

        rew += r
        if eo:
            break

    behavior = list(ENV.agent_pos)
    fitness = r

    return (behavior, (fitness,), info)


def evaluate_bipedal(individual):
    """Evaluates an individual: computes its value in the behavior descriptor space,
    and its fitness value.
    Bipedal walker

    Args:
        individual (Individual): an individual

    Returns:
        tuple: tuple of behavior (list) and fitness(tuple)
    """
    global ENV
    inf = {}

    ENV.reset()

    action = [0, 0, 0, 0]
    eo = False
    count = 0
    reward = 0
    behavior = []
    initial_height = ENV.hull.position[1]
    mean_diff_height = 0

    # CONTROLLER
    individual = np.array(individual)
    controller = utils.NeuralAgentNumpy(14, 4, n_hidden_layers=1, n_neurons_per_hidden=6)
    controller.set_weights(ind)
    while not eo and count <= 500:
        count += 1
        if DISPLAY:
            ENV.render()
        # apply previously chosen action
        o, r, eo, info = ENV.step(action)
        reward += r

        # stats
        mean_diff_height += initial_height - ENV.hull.position[1]

        action = controller.choose_action(o[:14])
    
    mean_diff_height = mean_diff_height / count
    behavior.append(mean_diff_height)

    x_final_pos = ENV.hull.position[0] / 200
    behavior.append(x_final_pos)

    return (behavior, (reward,), inf)


if __name__ == "__main__":
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 25

    if SIMPLE_RUN:
        size = MEDIUM_SIZE
    else:
        size = BIGGER_SIZE

    plt.rc('font', size=size, weight='bold')          # controls default text sizes
    plt.rc('axes', titlesize=size, titleweight='bold')     # fontsize of the axes title
    plt.rc('axes', labelsize=size, labelweight='bold')    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=size)    # legend fontsize
    plt.rc('figure', titlesize=size, titleweight='bold')  # fontsize of the figure title

    if ENV_NAME == 'maze':
        evaluate_individual = evaluate_maze

    if ENV_NAME == 'slime':
        evaluate_individual = evaluate_slime
    
    if ENV_NAME == 'beam':
        evaluate_individual = evaluate_beam
    
    if ENV_NAME == 'grid':
        evaluate_individual = evaluate_grid

    if ENV_NAME == 'bipedal':
        evaluate_individual = evaluate_bipedal

    if not ARCHIVE_ANALYSIS:
        if SIMPLE_RUN:
            archive_strat = 'least_novel'
            
            pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE, BD_BOUNDS,
                                                                 mini=MINI, archive_limit_size=None,
                                                                 plot=PLOT, algo_type='ns_rand', nb_gen=GEN,
                                                                 parallelize=PARALLELIZE, bound_genotype=1,
                                                                 measures=True, pop_size=POP_SIZE, nb_cells=NB_CELLS)

            if PLOT:
                if ENV_NAME == 'maze':
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
                
                if ENV_NAME == 'slime':
                    archive_behavior = np.array([ind.behavior_descriptor.values for ind in archive])
                    pop_behavior = np.array([ind.behavior_descriptor.values for ind in pop])
                    hof_behavior = np.array([ind.behavior_descriptor.values for ind in hof])
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.set(title='Final Archive', xlabel='Game duration', ylabel='Mean distance btw player and ball')
                    ax.scatter(archive_behavior[:, 0], archive_behavior[:, 1], color='red', label='Archive')
                    ax.scatter(pop_behavior[:, 0], pop_behavior[:, 1], color='blue', label='Population')
                    ax.scatter(hof_behavior[:, 0], hof_behavior[:, 1], color='green', label='Hall of Fame')
                    plt.legend()
                    plt.savefig('final_behavior.png')
                    fig = info['figure']
                    fig.savefig('exploration_slime.png')

                if ENV_NAME == 'bipedal':
                    archive_behavior = np.array([ind.behavior_descriptor.values for ind in archive])
                    pop_behavior = np.array([ind.behavior_descriptor.values for ind in pop])
                    hof_behavior = np.array([ind.behavior_descriptor.values for ind in hof])
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.set(title='Final Archive', xlabel='Mean height difference', ylabel='Final x position')
                    ax.scatter(archive_behavior[:, 0], archive_behavior[:, 1], color='red', label='Archive')
                    ax.scatter(pop_behavior[:, 0], pop_behavior[:, 1], color='blue', label='Population')
                    ax.scatter(hof_behavior[:, 0], hof_behavior[:, 1], color='green', label='Hall of Fame')
                    plt.legend()
                    plt.savefig('final_behavior.png')
                    fig = info['figure']
                    fig.savefig('exploration_bipedal.png')

            if SHOW_HOF:
                DISPLAY = True
                for ind in hof:
                    print(ind.behavior_descriptor.values)
                    evaluate_individual(ind)
                
            plt.show()
        else:
            # ############################### ANALYSE IMPORTANCE OF ARCHIVE #####################################
            fig, ax = plt.subplots(2, 1, figsize=(20, 15))
            # adding a run for classic ns
            coverages = []
            uniformities = []
            for i in range(N_EXP):
                pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE,
                                                                     BD_BOUNDS,
                                                                     mini=MINI, archive_limit_size=None,
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
            sig_cov = [mean_cov - np.std(coverages, 0), mean_cov + np.std(coverages, 0)]
            mean_uni = np.mean(uniformities, 0)
            std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
            sig_uni = [mean_uni - np.std(uniformities, 0), mean_uni + np.std(uniformities, 0)]

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
                                                                     mini=MINI, archive_limit_size=None, nb_gen=GEN,
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
            sig_cov = [mean_cov - np.std(coverages, 0), mean_cov + np.std(coverages, 0)]
            mean_uni = np.mean(uniformities, 0)
            std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
            sig_uni = [mean_uni - np.std(uniformities, 0), mean_uni + np.std(uniformities, 0)]

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
                                                                     mini=MINI, archive_limit_size=None, nb_gen=GEN,
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
            sig_cov = [mean_cov - np.std(coverages, 0), mean_cov + np.std(coverages, 0)]
            mean_uni = np.mean(uniformities, 0)
            std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
            sig_uni = [mean_uni - np.std(uniformities, 0), mean_uni + np.std(uniformities, 0)]

            ax[0].plot(mean_cov, label='random search', lw=2, color='orange')
            ax[0].fill_between(list(range(GEN)), sig_cov[0], sig_cov[1], facecolor='orange', alpha=0.5)
            ax[1].plot(mean_uni, label='random search', lw=2, color='orange')
            ax[1].fill_between(list(range(GEN)), sig_uni[0], sig_uni[1], facecolor='orange', alpha=0.5)

            # generating the plot
            ax[0].set_xlabel("Generations", labelpad=15, color="#333533")
            ax[1].set_xlabel("Generations", labelpad=15, color="#333533")
            ax[0].set_ylabel("Mean coverage", labelpad=15, color="#333533")
            ax[0].set_facecolor("#ffebb8")
            ax[0].legend(loc=4)
            ax[1].set_facecolor("#ffebb8")
            ax[1].set_ylabel("Mean uniformity", labelpad=15, color="#333533")
            ax[1].legend(loc=2)

            fig.savefig('archive_importance_maze.png')
            if PLOT:
                plt.show()
    else:
        if not NOVELTY_ANALYSIS:
            # ################################## ARCHIVE MANAGEMENT ANALYSIS ####################################
            possible_strats = ['random', 'least_novel', 'oldest', 'grid', 'grid_density', 'gmm', 'newest',
                               'least_novel_iter']
            colors = ['blue', 'red', 'yellow', 'green', 'pink', 'brown', 'purple', '#92F680']
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
                                                                         mini=MINI, archive_limit_size=ARCHIVE_LIMIT,
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
                sig_cov = [mean_cov - np.std(coverages, 0), mean_cov + np.std(coverages, 0)]
                mean_uni = np.mean(uniformities, 0)
                std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
                sig_uni = [mean_uni - np.std(uniformities, 0), mean_uni + np.std(uniformities, 0)]
                mean_arch_cov = np.mean(arch_coverages, 0)
                sig_arch_cov = [mean_arch_cov - np.std(arch_coverages, 0), mean_arch_cov + np.std(arch_coverages, 0)]
                std_arch_cov = [np.percentile(arch_coverages, 25, 0), np.percentile(arch_coverages, 75, 0)]
                mean_arch_uni = np.mean(arch_uniformities, 0)
                std_arch_uni = [np.percentile(arch_uniformities, 25, 0), np.percentile(arch_uniformities, 75, 0)]
                sig_arch_uni = [mean_arch_uni - np.std(arch_uniformities, 0),
                                mean_arch_uni + np.std(arch_uniformities, 0)]
                mean_rk_sim = np.mean(rk_sim, 0)

                ax[0].plot(mean_cov, label=archive_strat, lw=2, color=colors[s])
                ax[0].fill_between(list(range(GEN)), sig_cov[0], sig_cov[1], facecolor=colors[s], alpha=0.5)
                ax[1].plot(mean_uni, label=archive_strat, lw=2, color=colors[s])
                ax[1].fill_between(list(range(GEN)), sig_uni[0], sig_uni[1], facecolor=colors[s], alpha=0.5)
                ax[2].plot(mean_rk_sim, label=archive_strat, lw=2, color=colors[s])
                ax_2[0].plot(mean_arch_cov, label=archive_strat, lw=2, color=colors[s])
                ax_2[0].fill_between(list(range(GEN)), sig_arch_cov[0], sig_arch_cov[1], facecolor=colors[s], alpha=0.5)
                ax_2[1].plot(mean_arch_uni, label=archive_strat, lw=2, color=colors[s])
                ax_2[1].fill_between(list(range(GEN)), sig_arch_uni[0], sig_arch_uni[1], facecolor=colors[s], alpha=0.5)
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
                                                                     mini=MINI, archive_limit_size=None,
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
            sig_cov = [mean_cov - np.std(coverages, 0), mean_cov + np.std(coverages, 0)]
            mean_uni = np.mean(uniformities, 0)
            std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
            sig_uni = [mean_uni - np.std(uniformities, 0), mean_uni + np.std(uniformities, 0)]
            mean_arch_cov = np.mean(arch_coverages, 0)
            sig_arch_cov = [mean_arch_cov - np.std(arch_coverages, 0), mean_arch_cov + np.std(arch_coverages, 0)]
            std_arch_cov = [np.percentile(arch_coverages, 25, 0), np.percentile(arch_coverages, 75, 0)]
            mean_arch_uni = np.mean(arch_uniformities, 0)
            std_arch_uni = [np.percentile(arch_uniformities, 25, 0), np.percentile(arch_uniformities, 75, 0)]
            sig_arch_uni = [mean_arch_uni - np.std(arch_uniformities, 0),
                            mean_arch_uni + np.std(arch_uniformities, 0)]
            mean_rk_sim = np.mean(rk_sim, 0)

            ax[0].plot(mean_cov, label='no archive limit', lw=2, color='gray')
            ax[0].fill_between(list(range(GEN)), sig_cov[0], sig_cov[1], facecolor='gray', alpha=0.5)
            ax[1].plot(mean_uni, label='no archive limit', lw=2, color='gray')
            ax[1].fill_between(list(range(GEN)), sig_uni[0], sig_uni[1], facecolor='gray', alpha=0.5)
            ax_2[0].plot(mean_arch_cov, label='no archive limit', lw=2, color='gray')
            ax_2[0].fill_between(list(range(GEN)), sig_arch_cov[0], sig_arch_cov[1], facecolor='gray', alpha=0.5)
            ax_2[1].plot(mean_arch_uni, label='no archive limit', lw=2, color='gray')
            ax_2[1].fill_between(list(range(GEN)), sig_arch_uni[0], sig_arch_uni[1], facecolor='gray', alpha=0.5)

            # adding a run for random search
            coverages = []
            arch_coverages = []
            uniformities = []
            arch_uniformities = []
            rk_sim = []
            for i in range(N_EXP):
                pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE,
                                                                     BD_BOUNDS,
                                                                     mini=MINI, archive_limit_size=None,
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
            sig_cov = [mean_cov - np.std(coverages, 0), mean_cov + np.std(coverages, 0)]
            mean_uni = np.mean(uniformities, 0)
            std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
            sig_uni = [mean_uni - np.std(uniformities, 0), mean_uni + np.std(uniformities, 0)]
            mean_arch_cov = np.mean(arch_coverages, 0)
            sig_arch_cov = [mean_arch_cov - np.std(arch_coverages, 0), mean_arch_cov + np.std(arch_coverages, 0)]
            std_arch_cov = [np.percentile(arch_coverages, 25, 0), np.percentile(arch_coverages, 75, 0)]
            mean_arch_uni = np.mean(arch_uniformities, 0)
            std_arch_uni = [np.percentile(arch_uniformities, 25, 0), np.percentile(arch_uniformities, 75, 0)]
            sig_arch_uni = [mean_arch_uni - np.std(arch_uniformities, 0),
                            mean_arch_uni + np.std(arch_uniformities, 0)]
            mean_rk_sim = np.mean(rk_sim, 0)

            ax[0].plot(mean_cov, label='random search', lw=2, color='orange')
            ax[0].fill_between(list(range(GEN)), sig_cov[0], sig_cov[1], facecolor='orange', alpha=0.5)
            ax[1].plot(mean_uni, label='random search', lw=2, color='orange')
            ax[1].fill_between(list(range(GEN)), sig_uni[0], sig_uni[1], facecolor='orange', alpha=0.5)
            ax_2[0].plot(mean_arch_cov, label='random search', lw=2, color='orange')
            ax_2[0].fill_between(list(range(GEN)), sig_arch_cov[0], sig_arch_cov[1], facecolor='orange', alpha=0.5)
            ax_2[1].plot(mean_arch_uni, label='random search', lw=2, color='orange')
            ax_2[1].fill_between(list(range(GEN)), sig_arch_uni[0], sig_arch_uni[1], facecolor='orange', alpha=0.5)

            # adding a run for fitness ea
            coverages = []
            arch_coverages = []
            uniformities = []
            arch_uniformities = []
            rk_sim = []
            for i in range(N_EXP):
                pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE,
                                                                     BD_BOUNDS,
                                                                     mini=MINI, archive_limit_size=None,
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
            sig_cov = [mean_cov - np.std(coverages, 0), mean_cov + np.std(coverages, 0)]
            mean_uni = np.mean(uniformities, 0)
            std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
            sig_uni = [mean_uni - np.std(uniformities, 0), mean_uni + np.std(uniformities, 0)]
            mean_arch_cov = np.mean(arch_coverages, 0)
            sig_arch_cov = [mean_arch_cov - np.std(arch_coverages, 0), mean_arch_cov + np.std(arch_coverages, 0)]
            std_arch_cov = [np.percentile(arch_coverages, 25, 0), np.percentile(arch_coverages, 75, 0)]
            mean_arch_uni = np.mean(arch_uniformities, 0)
            std_arch_uni = [np.percentile(arch_uniformities, 25, 0), np.percentile(arch_uniformities, 75, 0)]
            sig_arch_uni = [mean_arch_uni - np.std(arch_uniformities, 0),
                            mean_arch_uni + np.std(arch_uniformities, 0)]
            mean_rk_sim = np.mean(rk_sim, 0)

            ax[0].plot(mean_cov, label='fitness ea', lw=2, color='cyan')
            ax[0].fill_between(list(range(GEN)), sig_cov[0], sig_cov[1], facecolor='cyan', alpha=0.5)
            ax[1].plot(mean_uni, label='fitness ea', lw=2, color='cyan')
            ax[1].fill_between(list(range(GEN)), sig_uni[0], sig_uni[1], facecolor='cyan', alpha=0.5)
            ax_2[0].plot(mean_arch_cov, label='fitness ea', lw=2, color='cyan')
            ax_2[0].fill_between(list(range(GEN)), sig_arch_cov[0], sig_arch_cov[1], facecolor='cyan', alpha=0.5)
            ax_2[1].plot(mean_arch_uni, label='fitness ea', lw=2, color='cyan')
            ax_2[1].fill_between(list(range(GEN)), sig_arch_uni[0], sig_arch_uni[1], facecolor='cyan', alpha=0.5)

            # generating the plot
            ax[0].set_xlabel("Generations", labelpad=15, color="#333533")
            ax[1].set_xlabel("Generations", labelpad=15, color="#333533")
            ax[2].set_xlabel("Iterations of reduction of archive", labelpad=15, color="#333533")
            ax[0].set_ylabel("Mean coverage", labelpad=15, color="#333533")
            ax[0].set_facecolor("#ffebb8")
            ax[0].legend(loc=4)
            ax[1].set_facecolor("#ffebb8")
            ax[1].set_ylabel("Mean uniformity", labelpad=15, color="#333533")
            ax[1].legend(loc=2)
            ax[2].set_ylabel("Mean Kendall Tau similarity", labelpad=15, color="#333533")
            ax[2].set_facecolor("#ffebb8")
            ax[2].legend(loc=4)

            ax_2[0].set_xlabel("Generations", labelpad=15, color="#333533")
            ax_2[1].set_xlabel("Generations", labelpad=15, color="#333533")
            ax_2[2].set_xlabel("Iterations of reduction of archive", labelpad=15, color="#333533")
            ax_2[0].set_ylabel("Mean archive coverage", labelpad=15, color="#333533")
            ax_2[0].set_facecolor("#ffebb8")
            ax_2[0].legend(loc=4)
            ax_2[1].set_facecolor("#ffebb8")
            ax_2[1].set_ylabel("Mean archive uniformity", labelpad=15, color="#333533")
            ax_2[1].legend(loc=2)
            ax_2[2].set_ylabel("Mean Kendall Tau similarity", labelpad=15, color="#333533")
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
                                                                         mini=MINI,
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
                sig_cov = [mean_cov - np.std(coverages, 0), mean_cov + np.std(coverages, 0)]
                mean_uni = np.mean(uniformities, 0)
                std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
                sig_uni = [mean_uni - np.std(uniformities, 0), mean_uni + np.std(uniformities, 0)]
                mean_arch_cov = np.mean(arch_coverages, 0)
                sig_arch_cov = [mean_arch_cov - np.std(arch_coverages, 0), mean_arch_cov + np.std(arch_coverages, 0)]
                std_arch_cov = [np.percentile(arch_coverages, 25, 0), np.percentile(arch_coverages, 75, 0)]
                mean_arch_uni = np.mean(arch_uniformities, 0)
                std_arch_uni = [np.percentile(arch_uniformities, 25, 0), np.percentile(arch_uniformities, 75, 0)]
                sig_arch_uni = [mean_arch_uni - np.std(arch_uniformities, 0),
                                mean_arch_uni + np.std(arch_uniformities, 0)]
                mean_rk_sim = np.mean(rk_sim, 0)

                ax[0].plot(mean_cov, label=archive_strat, lw=2, color=colors[s])
                ax[0].fill_between(list(range(GEN)), sig_cov[0], sig_cov[1], facecolor=colors[s], alpha=0.5)
                ax[1].plot(mean_uni, label=archive_strat, lw=2, color=colors[s])
                ax[1].fill_between(list(range(GEN)), sig_uni[0], sig_uni[1], facecolor=colors[s], alpha=0.5)
                ax[2].plot(mean_rk_sim, label=archive_strat, lw=2, color=colors[s])
                ax_2[0].plot(mean_arch_cov, label=archive_strat, lw=2, color=colors[s])
                ax_2[0].fill_between(list(range(GEN)), sig_arch_cov[0], sig_arch_cov[1], facecolor=colors[s], alpha=0.5)
                ax_2[1].plot(mean_arch_uni, label=archive_strat, lw=2, color=colors[s])
                ax_2[1].fill_between(list(range(GEN)), sig_arch_uni[0], sig_arch_uni[1], facecolor=colors[s], alpha=0.5)
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
                                                                     mini=MINI,
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
            sig_cov = [mean_cov - np.std(coverages, 0), mean_cov + np.std(coverages, 0)]
            mean_uni = np.mean(uniformities, 0)
            std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
            sig_uni = [mean_uni - np.std(uniformities, 0), mean_uni + np.std(uniformities, 0)]
            mean_arch_cov = np.mean(arch_coverages, 0)
            sig_arch_cov = [mean_arch_cov - np.std(arch_coverages, 0), mean_arch_cov + np.std(arch_coverages, 0)]
            std_arch_cov = [np.percentile(arch_coverages, 25, 0), np.percentile(arch_coverages, 75, 0)]
            mean_arch_uni = np.mean(arch_uniformities, 0)
            std_arch_uni = [np.percentile(arch_uniformities, 25, 0), np.percentile(arch_uniformities, 75, 0)]
            sig_arch_uni = [mean_arch_uni - np.std(arch_uniformities, 0),
                            mean_arch_uni + np.std(arch_uniformities, 0)]
            mean_rk_sim = np.mean(rk_sim, 0)

            ax[0].plot(mean_cov, label='no alteration', lw=2, color='gray')
            ax[0].fill_between(list(range(GEN)), sig_cov[0], sig_cov[1], facecolor='gray', alpha=0.5)
            ax[1].plot(mean_uni, label='no alteration', lw=2, color='gray')
            ax[1].fill_between(list(range(GEN)), sig_uni[0], sig_uni[1], facecolor='gray', alpha=0.5)
            ax_2[0].plot(mean_arch_cov, label='no alteration', lw=2, color='gray')
            ax_2[0].fill_between(list(range(GEN)), sig_arch_cov[0], sig_arch_cov[1], facecolor='gray', alpha=0.5)
            ax_2[1].plot(mean_arch_uni, label='no alteration', lw=2, color='gray')
            ax_2[1].fill_between(list(range(GEN)), sig_arch_uni[0], sig_arch_uni[1], facecolor='gray', alpha=0.5)

            # adding a run for random search
            coverages = []
            arch_coverages = []
            uniformities = []
            arch_uniformities = []
            rk_sim = []
            for i in range(N_EXP):
                pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE,
                                                                     BD_BOUNDS, altered_novelty=False,
                                                                     mini=MINI,
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
            sig_cov = [mean_cov - np.std(coverages, 0), mean_cov + np.std(coverages, 0)]
            mean_uni = np.mean(uniformities, 0)
            std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
            sig_uni = [mean_uni - np.std(uniformities, 0), mean_uni + np.std(uniformities, 0)]
            mean_arch_cov = np.mean(arch_coverages, 0)
            sig_arch_cov = [mean_arch_cov - np.std(arch_coverages, 0), mean_arch_cov + np.std(arch_coverages, 0)]
            std_arch_cov = [np.percentile(arch_coverages, 25, 0), np.percentile(arch_coverages, 75, 0)]
            mean_arch_uni = np.mean(arch_uniformities, 0)
            std_arch_uni = [np.percentile(arch_uniformities, 25, 0), np.percentile(arch_uniformities, 75, 0)]
            sig_arch_uni = [mean_arch_uni - np.std(arch_uniformities, 0),
                            mean_arch_uni + np.std(arch_uniformities, 0)]
            mean_rk_sim = np.mean(rk_sim, 0)

            ax[0].plot(mean_cov, label='random search', lw=2, color='orange')
            ax[0].fill_between(list(range(GEN)), sig_cov[0], sig_cov[1], facecolor='orange', alpha=0.5)
            ax[1].plot(mean_uni, label='random search', lw=2, color='orange')
            ax[1].fill_between(list(range(GEN)), sig_uni[0], sig_uni[1], facecolor='orange', alpha=0.5)
            ax_2[0].plot(mean_arch_cov, label='random search', lw=2, color='orange')
            ax_2[0].fill_between(list(range(GEN)), sig_arch_cov[0], sig_arch_cov[1], facecolor='orange', alpha=0.5)
            ax_2[1].plot(mean_arch_uni, label='random search', lw=2, color='orange')
            ax_2[1].fill_between(list(range(GEN)), sig_arch_uni[0], sig_arch_uni[1], facecolor='orange', alpha=0.5)

            # adding a run for fitness ea
            coverages = []
            arch_coverages = []
            uniformities = []
            arch_uniformities = []
            rk_sim = []
            for i in range(N_EXP):
                pop, archive, hof, info = noveltysearch.novelty_algo(evaluate_individual, INITIAL_GENOTYPE_SIZE,
                                                                     BD_BOUNDS, altered_novelty=False,
                                                                     mini=MINI,
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
            sig_cov = [mean_cov - np.std(coverages, 0), mean_cov + np.std(coverages, 0)]
            mean_uni = np.mean(uniformities, 0)
            std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
            sig_uni = [mean_uni - np.std(uniformities, 0), mean_uni + np.std(uniformities, 0)]
            mean_arch_cov = np.mean(arch_coverages, 0)
            sig_arch_cov = [mean_arch_cov - np.std(arch_coverages, 0), mean_arch_cov + np.std(arch_coverages, 0)]
            std_arch_cov = [np.percentile(arch_coverages, 25, 0), np.percentile(arch_coverages, 75, 0)]
            mean_arch_uni = np.mean(arch_uniformities, 0)
            std_arch_uni = [np.percentile(arch_uniformities, 25, 0), np.percentile(arch_uniformities, 75, 0)]
            sig_arch_uni = [mean_arch_uni - np.std(arch_uniformities, 0),
                            mean_arch_uni + np.std(arch_uniformities, 0)]
            mean_rk_sim = np.mean(rk_sim, 0)

            ax[0].plot(mean_cov, label='fitness ea', lw=2, color='cyan')
            ax[0].fill_between(list(range(GEN)), sig_cov[0], sig_cov[1], facecolor='cyan', alpha=0.5)
            ax[1].plot(mean_uni, label='fitness ea', lw=2, color='cyan')
            ax[1].fill_between(list(range(GEN)), sig_uni[0], sig_uni[1], facecolor='cyan', alpha=0.5)
            ax_2[0].plot(mean_arch_cov, label='fitness ea', lw=2, color='cyan')
            ax_2[0].fill_between(list(range(GEN)), sig_arch_cov[0], sig_arch_cov[1], facecolor='cyan', alpha=0.5)
            ax_2[1].plot(mean_arch_uni, label='fitness ea', lw=2, color='cyan')
            ax_2[1].fill_between(list(range(GEN)), sig_arch_uni[0], sig_arch_uni[1], facecolor='cyan', alpha=0.5)
            
            ax[0].set_xlabel("Generations", labelpad=15, color="#333533")
            ax[1].set_xlabel("Generations", labelpad=15, color="#333533")
            ax[2].set_xlabel("Iterations of novelty computation", labelpad=15, color="#333533")
            ax[0].set_ylabel("Mean coverage", labelpad=15, color="#333533")
            ax[0].set_facecolor("#ffebb8")
            ax[0].legend(loc=4)
            ax[1].set_facecolor("#ffebb8")
            ax[1].set_ylabel("Mean uniformity", labelpad=15, color="#333533")
            ax[1].legend(loc=2)
            ax[2].set_ylabel("Mean Kendall Tau similarity", labelpad=15, color="#333533")
            ax[2].set_facecolor("#ffebb8")
            ax[2].legend(loc=4)

            # generating the plot
            ax_2[0].set_xlabel("Generations", labelpad=15, color="#333533")
            ax_2[1].set_xlabel("Generations", labelpad=15, color="#333533")
            ax_2[2].set_xlabel("Iterations of novelty computation", labelpad=15, color="#333533")
            ax_2[0].set_ylabel("Mean archive coverage", labelpad=15, color="#333533")
            ax_2[0].set_facecolor("#ffebb8")
            ax_2[0].legend(loc=4)
            ax_2[1].set_facecolor("#ffebb8")
            ax_2[1].set_ylabel("Mean archive uniformity", labelpad=15, color="#333533")
            ax_2[1].legend(loc=2)
            ax_2[2].set_ylabel("Mean Kendall Tau similarity", labelpad=15, color="#333533")
            ax_2[2].set_facecolor("#ffebb8")
            ax_2[2].legend(loc=4)

            fig.savefig('full_analysis_novelty.png')
            fig_2.savefig('archive_analysis_novelty.png')
            if PLOT:
                plt.show()
