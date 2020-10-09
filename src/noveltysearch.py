import random
import numpy as np
import matplotlib.pyplot as plt
from deap import tools, base, creator
from scipy.spatial import cKDTree as KDTree


POP_SIZE = 100
HOF_SIZE = 10
K = 15  # number of nearest neighbours for novelty computation
ARCHIVE = 'random'  # random or novelty_based
ARCHIVE_PB = 0.2  # if ARCHIVE is random, probability to add individual to archive
ARCHIVE_NB = int(0.2 * POP_SIZE)  # if ARCHIVE is novelty_based, number of individuals added per generation

CXPB = 0.5  # probability with which two individuals are crossed
MUTPB = 0.2  # probability for mutating an individual
ABS_BOUND = 5  # absolute boundary for each dimension of individual genotype, can be None


def initialize_tool(initial_gen_size, min):
    """Initialize the toolbox

    Args:
        initial_gen_size (int): initial size of genotype
        min (bool): True if fitness to minimize, False if fitness to maximize

    Returns:
        Toolbox: the DEAP toolbox
    """
    # container for behavior descriptor
    creator.create('BehaviorDescriptor', list)
    # container for novelty
    creator.create('Novelty', base.Fitness, weights=(1.0,))
    # container for fitness
    if min:
        creator.create('Fit', base.Fitness, weights=(-1.0,))
    else:
        creator.create('Fit', base.Fitness, weights=(1.0,))

    # container for individual
    creator.create('Individual', list, behavior_descriptor=creator.BehaviorDescriptor,
                   novelty=creator.Novelty, fitness=creator.Fit)

    # create toolbox
    toolbox = base.Toolbox()

    # create function for individual initialization
    toolbox.register('init_ind', random.random)

    # create function for individual creation
    toolbox.register('individual', tools.initRepeat, creator.Individual,
                     toolbox.init_ind, initial_gen_size)

    # create function for population creation
    toolbox.register('population', tools.initRepeat, list, toolbox.individual, POP_SIZE)

    # create operators
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=0.1, indpb=0.3)
    toolbox.register('select', tools.selTournament, tournsize=3)
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
    for i in range(POP_SIZE):
        novelties.append(compute_average_distance(b_descriptors[i], k_tree))
    return novelties


def bound(offsprings):
    """Bound the individuals from the new population to the genotype constraints

    Args:
        offsprings (list): list of offsprings
    """
    for ind in offsprings:
        for i in range(len(ind)):
            if ind[i] > ABS_BOUND:
                ind[i] = ABS_BOUND
            if ind[i] < -ABS_BOUND:
                ind[i] = ABS_BOUND


def operate_offsprings(offsprings, toolbox):
    """Applies crossover and mutation to the offsprings

    Args:
        offsprings (list): list of offsprings
        toolbox (Toolbox): DEAP's toolbox
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
    if ABS_BOUND is not None:
        bound(offsprings)


def gen_plot(mean_hist, min_hist, max_hist, arch_size_hist):
    """Plotting

    Args:
        mean_hist (list): history of mean fitness
        min_hist (list): history of min fitness
        max_hist (list): history of max fitness
        arch_size_hist (list): history of archive size
    """
    mean_hist = np.array(mean_hist)
    min_hist = np.array(min_hist)

    # plot evolution
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set(title='Test with Rastrigin function', xlabel='Generations', ylabel='Fitness')
    ax.plot(mean_hist, label='Mean')
    ax.plot(min_hist, label='Min')
    ax.plot(max_hist, label='Max')

    plt.legend()

    # plot evolution
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set(title='Evolution of archive size', xlabel='Generations', ylabel='Archive size')
    ax.plot(arch_size_hist)


def novelty_algo(evaluate_individual, initial_gen_size, min=True, plot=False, nb_gen=100, algo_type='ns_nov'):

    creator, toolbox = initialize_tool(initial_gen_size, min)
    pop = toolbox.population()

    if algo_type == 'ns_nov':
        archive_type = 'novelty_based'
    if algo_type == 'ns_rand' or algo_type == 'classic_ea':
        archive_type = 'random'
    # plot initial population
    if plot:
        pop_n = np.array(pop)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set(title='Initial Population', xlabel='x1', ylabel='x2')
        ax.scatter(pop_n[:, 0], pop_n[:, 1])
        for i in range(len(pop)):
            ax.annotate(i, (pop_n[i, 0], pop_n[i, 1]))

    # initialize the archive
    archive = []

    # initialize the HOF
    hall_of_fame = tools.HallOfFame(HOF_SIZE)

    # initialize generation counter
    gen = 0

    # evaluate initial population
    evaluation_pop = list(map(evaluate_individual, pop))
    b_descriptors, fitnesses = map(list, zip(*evaluation_pop))

    # attribute fitness and behavior descriptors to individuals
    for ind, fit, bd in zip(pop, fitnesses, b_descriptors):
        ind.behavior_descriptor.values = bd
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
    if archive_type == 'novelty_based':
        # fill archive with the most novel individuals
        novel_n = np.array([nov[0] for nov in novelties])
        max_novelties_idx = np.argsort(-novel_n)[:ARCHIVE_NB]
        for i in max_novelties_idx:
            member = toolbox.clone(pop[i])
            archive.append(member)

    # keep track of stats
    mean_hist = []
    min_hist = []
    max_hist = []
    arch_size_hist = []

    # begin evolution
    while gen < nb_gen:
        # a new generation
        gen += 1
        print('--Generation %i --' % gen)

        if algo_type == 'ns_nov' or algo_type == 'ns_rand':
            # novelty search: selection on novelty
            offsprings = toolbox.select(pop, len(pop), fit_attr='novelty')  # references to selected individuals

        if algo_type == 'classic_ea':
            # classical EA: selection on fitness
            offsprings = toolbox.select(pop, len(pop), fit_attr='fitness')  # references to selected individuals
        
        offsprings = list(map(toolbox.clone, offsprings))  # clone selected indivduals
        operate_offsprings(offsprings, toolbox)  # crossover and mutation

        # evaluate the indivduals with an invalid fitness
        invalid_ind = [ind for ind in offsprings if not ind.fitness.valid]
        evaluation_pop = list(map(evaluate_individual, invalid_ind))
        inv_b_descriptors, inv_fitnesses = map(list, zip(*evaluation_pop))

        # attribute fitness and behavior descriptors to new individuals
        for ind, fit, bd in zip(invalid_ind, inv_fitnesses, inv_b_descriptors):
            ind.behavior_descriptor.values = bd
            ind.fitness.values = fit

        # full replacement of population
        pop[:] = offsprings

        # compute novelty for all current individuals (also old ones)
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
        if archive_type == 'novelty_based':
            # fill archive with the most novel individuals
            novel_n = np.array([nov[0] for nov in novelties])
            max_novelties_idx = np.argsort(-novel_n)[:ARCHIVE_NB]
            for i in max_novelties_idx:
                member = toolbox.clone(pop[i])
                archive.append(member)

        # update Hall of fame
        hall_of_fame.update(pop)

        # gather all the fitnesses in one list and compute stats
        fits = np.array([ind.fitness.values[0] for ind in pop])
        mean = np.mean(fits)
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
        gen_plot(mean_hist, min_hist, max_hist, arch_size_hist)

    # show all plots
    plt.show()

    return pop, archive, hall_of_fame
