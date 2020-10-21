import random
import numpy as np
import matplotlib.pyplot as plt
from deap import tools, base
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import distance
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


def initialize_tool(initial_gen_size, mini, pop_size, parallelize):
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

    # create function for population creation
    toolbox.register('population', tools.initRepeat, list, toolbox.individual, pop_size)

    # create operators
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=0.1, indpb=0.3)
    toolbox.register('select', tools.selTournament, tournsize=TOURNSIZE)
    toolbox.register('replace', tools.selBest)
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


def assess_novelties(pop, archive):
    """Compute novelties of current population

    Args:
        pop (list): list of current individuals
        archive (list): list of archive individuals

    Returns:
        list: list of novelties of current individuals
    """
    if not archive:
        # archive is empty --> only consider current population
        reference_pop = pop
    else:
        reference_pop = pop + archive
    # empty list for novelties
    novelties = []
    # extract all the behavior descriptors
    b_descriptors = [ind.behavior_descriptor.values for ind in reference_pop]
    k_tree = KDTree(b_descriptors)
    # compute novelty for current individuals
    for i in range(len(pop)):
        novelties.append(compute_average_distance(b_descriptors[i], k_tree))
    return novelties


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


def operate_offsprings(offsprings, toolbox, bound_genotype):
    """Applies crossover and mutation to the offsprings

    Args:
        offsprings (list): list of offsprings
        toolbox (Toolbox): DEAP's toolbox
        bound_genotype (float): absolute value bound of genotype values
    """
    # crossover
    for child1, child2 in zip(offsprings[::2], offsprings[1::2]):
        # child 1 has even index, child 2 has odd index
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            # both child1 and child2 are modified in place
            # del invalidates their fitness (new indivduals so unknown fitness)
            del child1.fitness.values
            del child2.fitness.values
    # mutation
    for mutant in offsprings:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # bound genotype to given constraints
    if bound_genotype is not None:
        bound(offsprings, bound_genotype)


def gen_plot(mean_hist, min_hist, max_hist, arch_size_hist, coverage_hist, uniformity_hist):
    """Plotting

    Args:
        mean_hist (list): history of mean fitness
        min_hist (list): history of min fitness
        max_hist (list): history of max fitness
        arch_size_hist (list): history of archive size
        coverage_hist (list): history of coverage of archive
        uniformity_hist (list): history of uniformity of archive
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
    ax.set(title='Evolution of archive size', xlabel='Generations', ylabel='Archive size')
    ax.plot(arch_size_hist)

    # plot evolution
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set(title='Evolution of selected metrics in archive', xlabel='Generations')
    ax.plot(coverage_hist, label='Coverage')
    ax.plot(uniformity_hist, label='Uniformity')
    plt.legend()


def add_to_grid(member, grid, cvt, measures):
    if measures:
        member_bd = member.behavior_descriptor.values
        grid_index = cvt.get_grid_index(member_bd)
        grid[grid_index] += 1


def novelty_algo(evaluate_individual, initial_gen_size, bd_bounds, mini=True, plot=False, nb_gen=100,
                 algo_type='ns_nov', bound_genotype=5, pop_size=30, parallelize=False,
                 measures=False):

    creator, toolbox = initialize_tool(initial_gen_size, mini, pop_size, parallelize)
    pop = toolbox.population()

    if algo_type == 'ns_nov':
        archive_type = 'novelty_based'
    if (algo_type == 'ns_rand' or algo_type == 'classic_ea') or algo_type == 'ns_rand_binary_removal':
        archive_type = 'random'

    # initialize the archive
    archive = []
    archive_nb = int(ARCHIVE_PB * OFFSPRING_NB_COEFF)

    # initialize the HOF
    hall_of_fame = tools.HallOfFame(HOF_SIZE)

    # initialize info dictionnary
    info = {}

    # initialize generation counter
    gen = 0

    grid = None
    cvt = None
    if measures:
        # initialize the CVT grid
        grid = np.zeros(NB_CELLS)
        cvt = utils.CVT(num_centroids=NB_CELLS, bounds=bd_bounds)

    # evaluate initial population
    evaluation_pop = list(toolbox.map(evaluate_individual, pop))
    b_descriptors, fitnesses, infos = map(list, zip(*evaluation_pop))

    # attribute fitness and behavior descriptors to individuals
    for ind, fit, bd, inf in zip(pop, fitnesses, b_descriptors, infos):
        ind.behavior_descriptor.values = bd
        ind.info.values = inf
        ind.fitness.values = fit

    novelties = assess_novelties(pop, archive)
    for ind, nov in zip(pop, novelties):
        ind.novelty.values = nov

    # fill archive
    if archive_type == 'random':
        # fill archive randomly
        for ind in pop:
            if random.random() < ARCHIVE_PB:
                member = toolbox.clone(ind)
                archive.append(member)
                add_to_grid(member, grid, cvt, measures)
    if archive_type == 'novelty_based':
        # fill archive with the most novel individuals
        novel_n = np.array([nov[0] for nov in novelties])
        max_novelties_idx = np.argsort(-novel_n)[:archive_nb]
        for i in max_novelties_idx:
            member = toolbox.clone(pop[i])
            archive.append(member)
            add_to_grid(member, grid, cvt, measures)

    # keep track of stats
    mean_hist = []
    min_hist = []
    max_hist = []
    arch_size_hist = []
    if measures:
        coverage_hist = []
        uniformity_hist = []

    # begin evolution
    while gen < nb_gen:
        # a new generation
        gen += 1
        print('--Generation %i --' % gen)

        if (algo_type == 'ns_nov' or algo_type == 'ns_rand') or algo_type == 'ns_rand_binary_removal':
            # novelty search: selection on novelty
            # references to selected individuals
            offsprings = toolbox.select(pop, int(pop_size * OFFSPRING_NB_COEFF), fit_attr='novelty')

        if algo_type == 'classic_ea':
            # classical EA: selection on fitness
            # references to selected individuals
            offsprings = toolbox.select(pop, int(pop_size * OFFSPRING_NB_COEFF), fit_attr='fitness')
        
        offsprings = list(map(toolbox.clone, offsprings))  # clone selected indivduals
        operate_offsprings(offsprings, toolbox, bound_genotype)  # crossover and mutation

        # evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offsprings if not ind.fitness.valid]
        evaluation_pop = list(toolbox.map(evaluate_individual, invalid_ind))
        inv_b_descriptors, inv_fitnesses, inv_infos = map(list, zip(*evaluation_pop))

        # attribute fitness and behavior descriptors to new individuals
        for ind, fit, bd, inf in zip(invalid_ind, inv_fitnesses, inv_b_descriptors, inv_infos):
            ind.behavior_descriptor.values = bd
            ind.info.values = inf
            ind.fitness.values = fit

        # current pool is old population + generated offsprings
        current_pool = pop + offsprings

        # compute novelty for all current individuals (also old ones)
        novelties = assess_novelties(current_pool, archive)
        for ind, nov in zip(current_pool, novelties):
            ind.novelty.values = nov

        # fill archive
        if archive_type == 'random':
            # fill archive randomly
            for ind in offsprings:
                if random.random() < ARCHIVE_PB:
                    member = toolbox.clone(ind)
                    archive.append(member)
                    add_to_grid(member, grid, cvt, measures)

        if archive_type == 'novelty_based':
            # fill archive with the most novel individuals
            offsprings_novelties = novelties[pop_size:]
            novel_n = np.array([nov[0] for nov in offsprings_novelties])
            max_novelties_idx = np.argsort(-novel_n)[:archive_nb]
            for i in max_novelties_idx:
                member = toolbox.clone(offsprings[i])
                archive.append(member)
                add_to_grid(member, grid, cvt, measures)

        # replacement: keep the most novel individuals
        pop[:] = toolbox.replace(current_pool, pop_size, fit_attr='novelty')

        if algo_type == 'ns_rand_binary_removal':
            # remove individuals that satisfy the binary goal
            for i, ind in enumerate(pop):
                if ind.info.values['binary goal']:
                    pop.pop(i)

        # update Hall of fame
        hall_of_fame.update(pop)

        # gather all the fitnesses in one list and compute stats
        fits = np.array([ind.fitness.values[0] for ind in pop])
        mean = np.mean(fits)
        if measures:
            # compute coverage
            coverage = np.count_nonzero(grid) / NB_CELLS
            coverage_hist.append(coverage)

            # compute uniformity
            P = grid[np.nonzero(grid)]
            P = P / np.sum(P)
            Q = np.ones(len(P)) / len(P)
            uniformity = distance.jensenshannon(P, Q)
            uniformity_hist.append(uniformity)

        # std = np.std(fits)
        # print("  Min %s" % np.min(fits))
        # print("  Max %s" % np.max(fits))
        # print("  Avg %s" % mean)
        # print("  Std %s" % std)
        arch_size_hist.append(len(archive))
        mean_hist.append(mean)
        min_hist.append(np.min(fits))
        max_hist.append(np.max(fits))

    if plot:
        gen_plot(mean_hist, min_hist, max_hist, arch_size_hist, coverage_hist, uniformity_hist)

    # show all plots
    plt.show()

    return pop, archive, hall_of_fame, info
