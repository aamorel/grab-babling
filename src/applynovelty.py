import gym
import noveltysearch
import time
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator
import utils
import math
import os
import json
import tqdm


DISPLAY = False
PARALLELIZE = False
GEN = 2
POP_SIZE = 20
ARCHIVE_LIMIT = None
NB_CELLS = 100
N_EXP = 10
ALGO = 'ns_rand'
PLOT = True
CASE = 'simple run'  # 'simple run', 'archive importance', 'novelty alteration', 'archive management'
ENV_NAME = 'maze'
SHOW_HOF = False
SAVE_ALL = False


def evaluate_slime(individual):
    """Evaluates an individual: computes its value in the behavior descriptor space,
    and its fitness value.
    Slime volley: modification of the env to make it deterministic
    extended 4 dim bd version

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
        jump_count = 0
        reward = 0
        distance_ball = 0
        distance_player = 0

        # CONTROLLER
        ind = np.array(individual)
        controller = utils.NeuralAgentNumpy(12, 3, n_hidden_layers=2, n_neurons_per_hidden=6)
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
            if action[2] == 1:
                jump_count += 1

            dist_to_ball = math.sqrt((o[0] - o[4])**2 + (o[1] - o[5])**2)
            distance_ball += dist_to_ball

            dist_to_player = math.sqrt((o[0] - o[8])**2 + (o[1] - o[9])**2)
            distance_player += dist_to_player

            if(DISPLAY):
                time.sleep(0.01)

        # use last info to compute behavior and fitness
        mean_distance_ball = distance_ball / (count * 4)
        
        mean_distance_player = distance_player / (count * 4)
        behavior_arr.append([count / 3000, mean_distance_player, mean_distance_ball, jump_count / count])

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
    controller.set_weights(individual)
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


def evaluate_billiard(individual):
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

    o = ENV.reset()

    eo = False
    count = 0
    reward = 0

    # CONTROLLER
    individual = np.array(individual)
    controller = utils.NeuralAgentNumpy(6, 2, n_hidden_layers=2, n_neurons_per_hidden=4)
    controller.set_weights(individual)
    while not eo and count <= 500:
        count += 1

        action = controller.choose_action(o)

        # apply previously chosen action
        o, r, eo, info = ENV.step(action)
        reward += r

    behavior = o[:2]

    return (behavior, (reward,), inf)


def evaluate_ant(individual):
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

    o = ENV.reset()

    eo = False
    count = 0
    reward = 0

    # CONTROLLER
    individual = np.array(individual)
    controller = utils.NeuralAgentNumpy(28, 8, n_hidden_layers=2, n_neurons_per_hidden=10)
    controller.set_weights(individual)
    while not eo and count <= 200:
        count += 1

        action = controller.choose_action(o)

        # apply previously chosen action
        o, r, eo, info = ENV.step(action)
        reward += r

    final_pos = ENV.robot.body_xyz
    behavior = [final_pos[0], final_pos[1]]

    return (behavior, (reward,), inf)


if ENV_NAME == 'maze':
    import gym_fastsim  # must still be imported
    BD_BOUNDS = [[0, 600], [0, 600]]
    INITIAL_GENOTYPE_SIZE = 12
    MINI = True
    EVALUATE_INDIVIDUAL = evaluate_maze
    BD_GENOTYPE = 1

if ENV_NAME == 'slime':
    import slimevolleygym  # must still be imported
    # global variable for the environment
    ENV = gym.make("SlimeVolley-v0")
    BD_BOUNDS = [[0, 1], [0, 1], [0, 1], [0, 1]]
    INITIAL_GENOTYPE_SIZE = 141
    # depends on the version of slimevolley (random or determinist)
    N_REPEAT = 1
    MINI = False
    EVALUATE_INDIVIDUAL = evaluate_slime
    BD_GENOTYPE = 1

if ENV_NAME == 'beam':
    import ballbeam_gym  # must still be imported
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
    EVALUATE_INDIVIDUAL = evaluate_beam
    BD_GENOTYPE = 1

if ENV_NAME == 'grid':
    import gym_minigrid  # must still be imported
    from gym_minigrid.wrappers import ImgObsWrapper  # must still be imported
    # create env
    ENV = ImgObsWrapper(gym.make('MiniGrid-Empty-8x8-v0'))
    BD_BOUNDS = [[0, 7], [0, 7]]
    NB_CELLS = 64
    INITIAL_GENOTYPE_SIZE = 11
    MINI = False
    EVALUATE_INDIVIDUAL = evaluate_grid
    BD_GENOTYPE = 1

if ENV_NAME == 'bipedal':
    # global variable for the environment
    ENV = gym.make('BipedalWalker-v3')
    BD_BOUNDS = [[-1, 1], [0, 1]]
    INITIAL_GENOTYPE_SIZE = 118
    MINI = False
    EVALUATE_INDIVIDUAL = evaluate_bipedal
    BD_GENOTYPE = 1

if ENV_NAME == 'billiard':
    import gym_billiard
    # global variable for the environment
    ENV = gym.make('Billiard-v0', max_steps=1000000)
    BD_BOUNDS = [[-1.5, 1.5], [-1.5, 1.5]]
    INITIAL_GENOTYPE_SIZE = 58
    MINI = False
    EVALUATE_INDIVIDUAL = evaluate_billiard
    BD_GENOTYPE = 1

if ENV_NAME == 'ant':
    # global variable for the environment
    ENV = utils.DeterministicPybulletAnt(render=DISPLAY, random_seed=0)
    BD_BOUNDS = [[-4, 4], [-4, 4]]
    INITIAL_GENOTYPE_SIZE = 488
    MINI = False
    EVALUATE_INDIVIDUAL = evaluate_ant
    BD_GENOTYPE = 1

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


def repeat_and_save(params):
    for i in tqdm.tqdm(range(N_EXP)):
        res = noveltysearch.novelty_algo(EVALUATE_INDIVIDUAL, INITIAL_GENOTYPE_SIZE,
                                         BD_BOUNDS, **params)
        
        pop, archive, hall_of_fame, details, figures, data, saved_ind = res

        i = 0
        while os.path.isfile('results/launch%i_details.json' % i):
            i += 1
        lauch_name = 'results/launch%i' % i

        # don't save some stuff
        if not SAVE_ALL:
            data['novelty distribution'] = None
            data['population genetic statistics'] = None
            data['offsprings genetic statistics'] = None

        with open(lauch_name + '_details.json', 'w') as fp:
            json.dump(details, fp, indent=4, sort_keys=True)

        with open(lauch_name + '_data.json', 'w') as fp:
            json.dump(data, fp, indent=4, sort_keys=True)


if __name__ == "__main__":

    if CASE == 'simple run':
        parameters = {'mini': MINI, 'archive_limit_size': ARCHIVE_LIMIT,
                      'plot': PLOT, 'algo_type': ALGO, 'nb_gen': GEN,
                      'parallelize': PARALLELIZE, 'bound_genotype': BD_GENOTYPE,
                      'measures': True, 'pop_size': POP_SIZE,
                      'nb_cells': NB_CELLS, 'save_ind_cond': 1, 'plot_gif': True}
        res = noveltysearch.novelty_algo(EVALUATE_INDIVIDUAL, INITIAL_GENOTYPE_SIZE,
                                         BD_BOUNDS, **parameters)
        pop, archive, hof, info, figures, data, saved_ind = res

        if PLOT:
            if ENV_NAME == 'maze':
                bds = []
                for offsprings in saved_ind:
                    for member in offsprings:
                        bds.append(member.behavior_descriptor.values)
                bds_arr = np.array(bds)

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
                ax.scatter(bds_arr[:, 0] / 3, bds_arr[:, 1] / 3, color='grey', label='Historic')
                if len(archive_behavior) > 0:
                    ax.scatter(archive_behavior[:, 0] / 3, archive_behavior[:, 1] / 3, color='red', label='Archive')
                ax.scatter(pop_behavior[:, 0] / 3, pop_behavior[:, 1] / 3, color='blue', label='Population')
                ax.scatter(hof_behavior[:, 0] / 3, hof_behavior[:, 1] / 3, color='green', label='Hall of Fame')
                plt.legend()

                fig.savefig('maze_exploration.png')

                gif = figures['gif']
                gif.save('maze_gif.gif')
            
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
                fig = figures['figure']
                fig.savefig('exploration_slime.png')

            if ENV_NAME == 'bipedal':
                bds = []
                for offsprings in saved_ind:
                    for member in offsprings:
                        bds.append(member.behavior_descriptor.values)
                bds_arr = np.array(bds)

                archive_behavior = np.array([ind.behavior_descriptor.values for ind in archive])
                pop_behavior = np.array([ind.behavior_descriptor.values for ind in pop])
                hof_behavior = np.array([ind.behavior_descriptor.values for ind in hof])
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.set(title='Final Archive', xlabel='Mean height difference', ylabel='Final x position')
                ax.scatter(bds_arr[:, 0], bds_arr[:, 1], color='grey', label='Historic')
                ax.scatter(archive_behavior[:, 0], archive_behavior[:, 1], color='red', label='Archive')
                ax.scatter(pop_behavior[:, 0], pop_behavior[:, 1], color='blue', label='Population')
                ax.scatter(hof_behavior[:, 0], hof_behavior[:, 1], color='green', label='Hall of Fame')
                ax.legend()
                fig.savefig('bipedal_exploration.png')

                gif = figures['gif']
                gif.save('bipedal_gif.gif')
            
            if ENV_NAME == 'billiard':
                archive_behavior = np.array([ind.behavior_descriptor.values for ind in archive])
                pop_behavior = np.array([ind.behavior_descriptor.values for ind in pop])
                hof_behavior = np.array([ind.behavior_descriptor.values for ind in hof])
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.set(title='Final Archive', xlabel='x', ylabel='y')
                ax.scatter(archive_behavior[:, 0], archive_behavior[:, 1], color='red', label='Archive')
                ax.scatter(pop_behavior[:, 0], pop_behavior[:, 1], color='blue', label='Population')
                ax.scatter(hof_behavior[:, 0], hof_behavior[:, 1], color='green', label='Hall of Fame')
                plt.legend()
                
            if ENV_NAME == 'ant':
                bds = []
                for offsprings in saved_ind:
                    for member in offsprings:
                        bds.append(member.behavior_descriptor.values)
                bds_arr = np.array(bds)

                archive_behavior = np.array([ind.behavior_descriptor.values for ind in archive])
                pop_behavior = np.array([ind.behavior_descriptor.values for ind in pop])
                hof_behavior = np.array([ind.behavior_descriptor.values for ind in hof])
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.set(title='Final Archive', xlabel='x', ylabel='y')
                ax.scatter(bds_arr[:, 0], bds_arr[:, 1], color='grey', label='Historic')
                ax.scatter(archive_behavior[:, 0], archive_behavior[:, 1], color='red', label='Archive')
                ax.scatter(pop_behavior[:, 0], pop_behavior[:, 1], color='blue', label='Population')
                ax.scatter(hof_behavior[:, 0], hof_behavior[:, 1], color='green', label='Hall of Fame')
                ax.legend()
                fig.savefig('ant_exploration.png')

                gif = figures['gif']
                gif.save('ant_gif.gif')

            plt.show()

        if SHOW_HOF:
            DISPLAY = True
            for ind in hof:
                print(ind.behavior_descriptor.values)
                EVALUATE_INDIVIDUAL(ind)
            
    if CASE == 'archive importance':

        parameters = {'mini': MINI, 'archive_limit_size': None,
                      'plot': False, 'algo_type': 'ns_rand', 'nb_gen': GEN,
                      'parallelize': True, 'bound_genotype': BD_GENOTYPE,
                      'measures': True, 'pop_size': POP_SIZE,
                      'nb_cells': NB_CELLS}
        
        # classic NS runs
        repeat_and_save(parameters)

        # no archive NS runs
        parameters['algo_type'] = 'ns_no_archive'
        repeat_and_save(parameters)

        # random search runs
        parameters['algo_type'] = 'random_search'
        repeat_and_save(parameters)

    if CASE == 'novelty alteration':

        parameters = {'mini': MINI, 'archive_limit_size': None,
                      'plot': False, 'algo_type': 'ns_rand', 'nb_gen': GEN,
                      'parallelize': True, 'bound_genotype': BD_GENOTYPE,
                      'measures': True, 'pop_size': POP_SIZE,
                      'nb_cells': NB_CELLS, 'altered_novelty': True}
        possible_degrees = [0.1, 0.5, 1, 5, 10, 20, 100]

        # alteration runs
        for degree in possible_degrees:
            parameters['alteration_degree'] = degree
            repeat_and_save(parameters)
        
        # classic ns runs
        parameters['altered_novelty'] = False
        repeat_and_save(parameters)

        # random search runs
        parameters['algo_type'] = 'random_search'
        repeat_and_save(parameters)

        # fitness ea runs
        parameters['algo_type'] = 'classic_ea'
        repeat_and_save(parameters)

    if CASE == 'archive management':

        parameters = {'mini': MINI, 'archive_limit_size': ARCHIVE_LIMIT,
                      'plot': False, 'algo_type': 'ns_rand', 'nb_gen': GEN,
                      'parallelize': True, 'bound_genotype': BD_GENOTYPE,
                      'measures': True, 'pop_size': POP_SIZE,
                      'nb_cells': NB_CELLS}
        possible_strats = ['random', 'least_novel', 'oldest', 'grid', 'grid_density', 'gmm', 'newest',
                           'least_novel_iter', 'most_novel']

        # alteration runs
        for strat in possible_strats:
            parameters['archive_limit_strat'] = strat
            repeat_and_save(parameters)
        
        # classic ns runs
        parameters['archive_limit_size'] = None
        repeat_and_save(parameters)

        # random search runs
        parameters['algo_type'] = 'random_search'
        repeat_and_save(parameters)

        # fitness ea runs
        parameters['algo_type'] = 'classic_ea'
        repeat_and_save(parameters)

    if CASE == 'most novel management':
        parameters = {'mini': MINI, 'archive_limit_size': ARCHIVE_LIMIT,
                      'plot': False, 'algo_type': 'ns_rand', 'nb_gen': GEN,
                      'parallelize': True, 'bound_genotype': BD_GENOTYPE,
                      'measures': True, 'pop_size': POP_SIZE,
                      'nb_cells': NB_CELLS}
        possible_strats = ['most_novel']

        # alteration runs
        for strat in possible_strats:
            parameters['archive_limit_strat'] = strat
            repeat_and_save(parameters)
