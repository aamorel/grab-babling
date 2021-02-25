import random
import numpy as np
import matplotlib.pyplot as plt
from deap import tools, base
from scoop import futures
import utils

creator = None


def set_creator(cr):
    global creator
    creator = cr


HOF_SIZE = 10  # number of individuals in hall of fame
NB_CELLS = 1000  # number of cells in measuring grid
K = 15  # number of nearest neighbours for novelty computation
ARCHIVE = 'random'  # random or novelty_based
ARCHIVE_PB = 0.2  # if ARCHIVE is random, probability to add individual to archive
# if ARCHIVE is novelty_based, proportion of individuals added per gen
CXPB = 0.5  # probability with which two individuals are crossed
MUTPB = 0.2  # probability for mutating an individual
TOURNSIZE = 10  # drives towards selective pressure
OFFSPRING_NB_COEFF = 0.5  # number of offsprings generated (coeff of pop length)
N_RAND = 200  # number of random initial individuals


def initialize_tool(initial_gen_size, mini, parallelize):
    """Initialize the toolbox

    Args:
        initial_gen_size (int): initial size of genotype
        min (bool): True if fitness to minimize, False if fitness to maximize
        pop_size (int): size of population
        parallelize (bool): True if evaluation of individuals must be parallelized

    Returns:
        Toolbox: the DEAP toolbox
    """
    global creator
    # create toolbox
    toolbox = base.Toolbox()

    if not parallelize:
        from deap import creator
        # container for behavior descriptor
        creator.create('BehaviorDescriptor', list)
        # container for info
        creator.create('Info', dict)
        # container for novelty
        creator.create('Novelty', base.Fitness, weights=(1.0,))
        # container for fitness
        if mini:
            creator.create('Fit', base.Fitness, weights=(-1.0,))
        else:
            creator.create('Fit', base.Fitness, weights=(1.0,))

        # container for individual
        creator.create('Individual', list, behavior_descriptor=creator.BehaviorDescriptor,
                       novelty=creator.Novelty, fitness=creator.Fit, info=creator.Info)

        # overwrite map function with normal map
        toolbox.register('map', map)
    else:
        # overwrite map function with scoop for parallelization
        toolbox.register('map', futures.map)

    # create function for individual initialization
    toolbox.register('init_ind', random.uniform, -1, 1)

    # create function for individual creation
    toolbox.register('individual', tools.initRepeat, creator.Individual,
                     toolbox.init_ind, initial_gen_size)

    # create operators
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=0.1, indpb=0.3)
    toolbox.register('select', tools.selBest)
    return creator, toolbox


def compute_average_distance(query, k_tree):
    """Finds K nearest neighbours and distances

    Args:
        query (int): index of the individual
        k_tree (KDTree): tree in the behavior descriptor space

    Returns:
        float: average distance to the K nearest neighbours
    """
    # find K nearest neighbours and distances
    neighbours = k_tree.query(query, range(2, K + 2))
    # compute mean distance
    avg_distance = np.mean(neighbours[0])
    return avg_distance,


def bound(offsprings, bound_genotype):
    """Bound the individuals from the new population to the genotype constraints

    Args:
        offsprings (list): list of offsprings
    """
    for ind in offsprings:
        for i in range(len(ind)):
            if ind[i] > bound_genotype:
                ind[i] = bound_genotype
            if ind[i] < -bound_genotype:
                ind[i] = bound_genotype


def gen_plot(mean_hist, min_hist, max_hist, coverage_hist):
    """Plotting

    Args:
        mean_hist (list): history of mean fitness
        min_hist (list): history of min fitness
        max_hist (list): history of max fitness
        coverage_hist (list): history of coverage of archive
   """
    mean_hist = np.array(mean_hist)
    min_hist = np.array(min_hist)

    # plot evolution
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set(title='Evolution of fitness', xlabel='Generations', ylabel='Fitness')
    ax.plot(mean_hist, label='Mean')
    ax.plot(min_hist, label='Min')
    ax.plot(max_hist, label='Max')
    plt.legend()

    # plot evolution
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set(title='Evolution of selected metrics in grid', xlabel='Generations')
    ax.plot(coverage_hist, label='Coverage')
    plt.legend()


def add_to_grid(member, grid, cvt, toolbox):

    # find the corresponding cell
    member_bd = member.behavior_descriptor.values
    grid_index = cvt.get_grid_index(member_bd)
    
    # get the competitor
    competitor = grid[grid_index]

    if competitor is None:
        # no competitor --> add to cell
        grid[grid_index] = member
    else:
        # select best of the two and place in the cell
        competition = [member, competitor]
        winner = toolbox.select(competition, 1, fit_attr='fitness')[0]
        grid[grid_index] = winner


def map_elites_algo(evaluate_individual, initial_gen_size, bd_bounds, mini=True, plot=False, nb_gen=1000,
                    bound_genotype=5, parallelize=False):

    creator, toolbox = initialize_tool(initial_gen_size, mini, parallelize)

    # initialize info dictionnary
    info = {}

    # initialize generation counter
    gen = 0
    
    # initialize the CVT grid
    grid = [None] * NB_CELLS
    cvt = utils.CVT(num_centroids=NB_CELLS, bounds=bd_bounds)

    # keep track of stats
    mean_hist = []
    min_hist = []
    max_hist = []
    coverage_hist = []

    # begin evolution
    while gen < nb_gen:
        gen += 1
        print('--Generation %i --' % gen)
        if gen <= N_RAND:
            # generate new random individual
            ind = toolbox.individual()
        
        else:
            # find a random parent in grid and mutate it
            found = False
            while not found:
                selected_index = random.randint(0, NB_CELLS - 1)
                parent = grid[selected_index]
                if parent is not None:
                    found = True
            ind = toolbox.clone(parent)
            toolbox.mutate(ind)
            # bound genotype to given constraints
            if bound_genotype is not None:
                bound([ind], bound_genotype)

        # evaluate new individual
        bd, fit, inf = evaluate_individual(ind)
        ind.behavior_descriptor.values = bd
        ind.info.values = inf
        ind.fitness.values = fit

        add_to_grid(ind, grid, cvt, toolbox)

        # gather all the fitnesses in one list and compute stats
        fits = np.array([ind.fitness.values[0] for ind in grid if ind is not None])
        mean = np.mean(fits)

        # compute coverage
        coverage = sum(ind is not None for ind in grid) / NB_CELLS
        coverage_hist.append(coverage)

        mean_hist.append(mean)
        min_hist.append(np.min(fits))
        max_hist.append(np.max(fits))

    if plot:
        gen_plot(mean_hist, min_hist, max_hist, coverage_hist)

    # show all plots
    plt.show()

    return grid, info
