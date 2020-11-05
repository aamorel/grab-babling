import random
import numpy as np
import matplotlib.pyplot as plt
from deap import tools, base
from scipy.spatial import cKDTree as KDTree
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
CXPB = 0.2  # probability with which two individuals are crossed
MUTPB = 0.8  # probability for mutating an individual
SIGMA = 0.1  # std of the mutation of one gene
# CXPB and MUTPB should sum up to 1
TOURNSIZE = 10  # drives towards selective pressure
OFFSPRING_NB_COEFF = 0.5  # number of offsprings generated (coeff of pop length)
MAX_MUT_PROB = 0.5  # if algo is ns_rand_keep_diversity, probability for the less diverse gene to be mutated

id_counter = 0  # each individual will have a unique id


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
        # container for genetic info
        creator.create('GenInfo', dict)
        # container for novelty
        creator.create('Novelty', base.Fitness, weights=(1.0,))
        # container for fitness
        if mini:
            creator.create('Fit', base.Fitness, weights=(-1.0,))
        else:
            creator.create('Fit', base.Fitness, weights=(1.0,))

        # container for individual
        creator.create('Individual', list, behavior_descriptor=creator.BehaviorDescriptor,
                       novelty=creator.Novelty, fitness=creator.Fit, info=creator.Info,
                       gen_info=creator.GenInfo)

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
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=SIGMA, indpb=0.3)
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
    # extract all the behavior descriptors ( [pop[0], ..., pop[n], archive[0], ..., archive[n]] )
    b_descriptors = [ind.behavior_descriptor.values for ind in reference_pop]
    # extract all the behavior descriptors that are not None to create the tree
    b_ds = [ind.behavior_descriptor.values for ind in reference_pop if ind.behavior_descriptor.values is not None]
    k_tree = KDTree(b_ds)
    # compute novelty for current individuals (loop only on the pop)
    for i in range(len(pop)):
        if b_descriptors[i] is not None:
            novelties.append(compute_average_distance(b_descriptors[i], k_tree))
        else:
            novelties.append(0)
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
    global id_counter
    for i, ind in enumerate(offsprings):

        if random.random() < CXPB and i < (len(offsprings) - 1):
            # crossover
            other_parent = toolbox.clone(offsprings[i + 1])  # in order to not modify the other parent
            parents_id = [ind.gen_info.values['id'], other_parent.gen_info.values['id']]
            toolbox.mate(ind, other_parent)
            # ind is modified in place
            del ind.fitness.values
            ind.gen_info.values['parent id'] = parents_id
        else:
            # mutation
            toolbox.mutate(ind)
            del ind.fitness.values
            ind.gen_info.values['parent id'] = ind.gen_info.values['id']

        # update new individual's genetic info
        ind.gen_info.values['id'] = id_counter
        id_counter += 1
        ind.gen_info.values['age'] = 0

    # bound genotype to given constraints
    if bound_genotype is not None:
        bound(offsprings, bound_genotype)


def genetic_stats(pop):
    """Computes the genetic standard deviation

    Args:
        pop (list): list of individuals

    Returns:
        list: array of same length as genotype containing the per gene std
    """
    gen_stats = [[] for _ in range(len(pop[0]))]
    for ind in pop:
        for i, gene in enumerate(ind):
            gen_stats[i].append(gene)
    gen_stats = np.array(gen_stats)
    std_genes = np.std(gen_stats, axis=1)
    return std_genes


def operate_offsprings_diversity(offsprings, toolbox, bound_genotype, pop):
    """Applies crossover and mutation to the offsprings
    Aims at keeping the population diversified by mutating the different genes based on their diversity
    inside the population: the less a gene is diverse in the population, the more it will be mutated.
    Args:
        offsprings (list): list of offsprings
        toolbox (Toolbox): DEAP's toolbox
        bound_genotype (float): absolute value bound of genotype values
        pop (list): current population
    """
    global id_counter

    std_genes = genetic_stats(pop)  # contains the std of each gene on the population
    # we want the probability of one gene to be mutated proportional to the inverse of the std
    mut_prob = np.reciprocal(std_genes + 0.0000001)
    mut_max = np.max(mut_prob)
    mut_prob = mut_prob / mut_max * MAX_MUT_PROB  # highest probability to be mutated is MAX_MUT_PROB

    for i, ind in enumerate(offsprings):

        if random.random() < CXPB and i < (len(offsprings) - 1):
            # crossover
            other_parent = toolbox.clone(offsprings[i + 1])  # in order to not modify the other parent
            parents_id = [ind.gen_info.values['id'], other_parent.gen_info.values['id']]
            toolbox.mate(ind, other_parent)
            # ind is modified in place
            del ind.fitness.values
            ind.gen_info.values['parent id'] = parents_id
        else:
            # custom mutation
            for j, gene in enumerate(ind):
                if random.random() < mut_prob[j]:
                    # mutate the gene
                    ind[j] = gene + random.gauss(0, SIGMA)
            del ind.fitness.values
            ind.gen_info.values['parent id'] = ind.gen_info.values['id']

        # update new individual's genetic info
        ind.gen_info.values['id'] = id_counter
        id_counter += 1
        ind.gen_info.values['age'] = 0

    # bound genotype to given constraints
    if bound_genotype is not None:
        bound(offsprings, bound_genotype)


def gen_plot(mean_hist, min_hist, max_hist, arch_size_hist, coverage_hist, uniformity_hist,
             mean_age_hist, max_age_hist, run_name):
    """Plotting

    Args:
        mean_hist (list): history of mean population fitness
        min_hist (list): history of min population fitness
        max_hist (list): history of max population fitness
        arch_size_hist (list): history of archive size
        coverage_hist (list): history of coverage of archive
        uniformity_hist (list): history of uniformity of archive
        mean_age_hist (list): history of mean age of population
        max_age_hist (list): history of max age of population
        run_name (String): path of the run folder to save the figure
   """
    mean_hist = np.array(mean_hist)
    min_hist = np.array(min_hist)

    # plot evolution
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    ax[0][0].set(title='Evolution of fitness in population', xlabel='Generations', ylabel='Fitness')
    ax[0][0].plot(mean_hist, label='Mean')
    ax[0][0].plot(min_hist, label='Min')
    ax[0][0].plot(max_hist, label='Max')
    ax[0][0].legend()

    # plot evolution
    ax[1][0].set(title='Evolution of age in population', xlabel='Generations', ylabel='Age')
    ax[1][0].plot(mean_age_hist, label='Mean')
    ax[1][0].plot(max_age_hist, label='Max')
    ax[1][0].legend()

    # plot evolution
    ax[0][1].set(title='Evolution of archive size', xlabel='Generations', ylabel='Archive size')
    ax[0][1].plot(arch_size_hist)

    # plot evolution
    ax[1][1].set(title='Evolution of selected metrics in archive', xlabel='Generations')
    ax[1][1].plot(coverage_hist, label='Coverage')
    ax[1][1].plot(uniformity_hist, label='Uniformity')
    ax[1][1].legend()

    if run_name is not None:
        plt.savefig(run_name + 'novelty_search_plots.png')


def add_to_grid(member, grid, cvt, measures):
    if measures:
        member_bd = member.behavior_descriptor.values
        grid_index = cvt.get_grid_index(member_bd)
        grid[grid_index] += 1


def novelty_algo(evaluate_individual_list, initial_gen_size, bd_bounds_list, mini=True, plot=False, nb_gen=100,
                 algo_type='ns_nov', bound_genotype=5, pop_size=30, parallelize=False,
                 measures=False, run_name=None, choose_evaluate=None):

    global id_counter

    creator, toolbox = initialize_tool(initial_gen_size, mini, pop_size, parallelize)

    # initialize evaluate_individual function
    if algo_type == 'ns_rand_multi_bd':
        evaluate_individual = evaluate_individual_list[0]
        bd_bounds = bd_bounds_list[0]
        eval_index = 0
        info_multi_bd = {}
        info_multi_bd['changed'] = False

    else:
        evaluate_individual = evaluate_individual_list
        bd_bounds = bd_bounds_list

    # initialize population
    pop = toolbox.population()
    for ind in pop:
        ind.gen_info.values = {}
        # attribute id to all individuals
        ind.gen_info.values['id'] = id_counter
        id_counter += 1
        # attribute -1 to parent id (convention for initial individuals)
        ind.gen_info.values['parent id'] = -1
        # attribute age of 0
        ind.gen_info.values['age'] = 0

    if algo_type == 'ns_nov':
        archive_type = 'novelty_based'
    else:
        archive_type = 'random'

    # initialize the archive
    archive = []
    archive_nb = int(ARCHIVE_PB * OFFSPRING_NB_COEFF)

    # initialize the HOF
    hall_of_fame = tools.HallOfFame(HOF_SIZE)

    # initialize details dictionnary
    details = {}

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

    # fill archive with clones of population
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
    mean_age_hist = []
    max_age_hist = []
    gen_stat_hist = []
    gen_stat_hist_off = []
    if measures:
        coverage_hist = []
        uniformity_hist = []

    # begin evolution
    while gen < nb_gen:
        # a new generation
        gen += 1
        print('--Generation %i --' % gen)

        if algo_type == 'classic_ea':
            # classical EA: selection on fitness
            # references to selected individuals
            offsprings = toolbox.select(pop, int(pop_size * OFFSPRING_NB_COEFF), fit_attr='fitness')

        else:
            # novelty search: selection on novelty
            # references to selected individuals
            offsprings = toolbox.select(pop, int(pop_size * OFFSPRING_NB_COEFF), fit_attr='novelty')
        
        offsprings = list(map(toolbox.clone, offsprings))  # clone selected indivduals
        # for now, offsprings are parents --> they keep the genetic information
        if algo_type == 'ns_rand_keep_diversity':
            operate_offsprings_diversity(offsprings, toolbox, bound_genotype,
                                         pop)  # crossover and mutation
        else:
            operate_offsprings(offsprings, toolbox, bound_genotype)  # crossover and mutation
        # now, offsprings have their correct genetic information

        # current pool is old population + generated offsprings
        current_pool = pop + offsprings

        # evaluate the individuals with an invalid fitness
        # done for the whole current pool and not just the offsprings in case the evaluation
        # function has changed
        invalid_ind = [ind for ind in current_pool if not ind.fitness.valid]
        evaluation_pop = list(toolbox.map(evaluate_individual, invalid_ind))
        inv_b_descriptors, inv_fitnesses, inv_infos = map(list, zip(*evaluation_pop))

        # attribute fitness and behavior descriptors to new individuals
        for ind, fit, bd, inf in zip(invalid_ind, inv_fitnesses, inv_b_descriptors, inv_infos):
            ind.behavior_descriptor.values = bd  # can be None in the multi_bd case
            ind.info.values = inf
            ind.fitness.values = fit

        # compute novelty for all current individuals (also old ones)
        novelties = assess_novelties(current_pool, archive)
        for ind, nov in zip(current_pool, novelties):
            ind.novelty.values = nov
        # an individual with bd = None will have 0 novelty

        # fill archive
        if archive_type == 'random':
            # fill archive randomly
            for ind in offsprings:
                if random.random() < ARCHIVE_PB and ind.behavior_descriptor.values is not None:
                    member = toolbox.clone(ind)
                    archive.append(member)
                    add_to_grid(member, grid, cvt, measures)

        if archive_type == 'novelty_based':
            # fill archive with the most novel individuals
            offsprings_novelties = novelties[pop_size:]
            novel_n = np.array([nov[0] for nov in offsprings_novelties])
            max_novelties_idx = np.argsort(-novel_n)[:archive_nb]
            for i in max_novelties_idx:
                if offsprings[i].behavior_descriptor.values is not None:
                    member = toolbox.clone(offsprings[i])
                    archive.append(member)
                    add_to_grid(member, grid, cvt, measures)

        # replacement: keep the most novel individuals
        pop[:] = toolbox.replace(current_pool, pop_size, fit_attr='novelty')

        if algo_type == 'ns_rand_binary_removal':
            # remove individuals that satisfy the binary goal
            # they can still be in the archive
            for i, ind in enumerate(pop):
                if ind.info.values['binary goal']:
                    pop.pop(i)

        # increment age of the individuals in the population
        for ind in pop:
            ind.gen_info.values['age'] += 1

        # update Hall of fame
        hall_of_fame.update(pop)

        # compute genetic statistics of the population
        genes_std = genetic_stats(pop)
        gen_stat_hist.append(genes_std)

        # compute genetic statistics of the offsprings
        genes_std = genetic_stats(offsprings)
        gen_stat_hist_off.append(genes_std)

        # gather all the fitnesses in one list and compute stats
        fits = np.array([ind.fitness.values[0] for ind in pop])
        mean_fit = np.mean(fits)

        # gather all the ages in one list and compute stats
        ages = np.array([ind.gen_info.values['age'] for ind in pop])
        mean_age = np.mean(ages)
        if measures:
            # compute coverage
            coverage = np.count_nonzero(grid) / NB_CELLS
            coverage_hist.append(coverage)

            # compute uniformity
            uniformity = utils.compute_uniformity(grid)
            uniformity_hist.append(uniformity)

        arch_size_hist.append(len(archive))
        mean_hist.append(mean_fit)
        min_hist.append(np.min(fits))
        max_hist.append(np.max(fits))
        max_age_hist.append(np.max(ages))
        mean_age_hist.append(mean_age)

        if algo_type == 'ns_rand_multi_bd':
            # novelty search expects a list of evaluation functions and a list of bd_bounds
            # and a choose_evaluate functions which chooses which evaluation function to use
            # based on the info_multi_bd dictionnary (returns an index)
            info_multi_bd['coverage'] = coverage
            
            old_eval_index = eval_index
            eval_index = choose_evaluate(info_multi_bd)
            evaluate_individual = evaluate_individual_list[eval_index]
            bd_bounds = bd_bounds_list[eval_index]
            if old_eval_index != eval_index:
                # behavior descriptor has changed
                # add info
                info_multi_bd['changed'] = True
                
                # empty the archive
                archive = []

                # set the current population to be re-evaluated
                for ind in pop:
                    del ind.fitness.values

                if measures:
                    # re-initialize the CVT grid
                    grid = np.zeros(NB_CELLS)
                    cvt = utils.CVT(num_centroids=NB_CELLS, bounds=bd_bounds)
    
    details['population genetic statistics'] = gen_stat_hist
    details['offsprings genetic statistics'] = gen_stat_hist_off

    if plot:
        gen_plot(mean_hist, min_hist, max_hist, arch_size_hist, coverage_hist, uniformity_hist,
                 mean_age_hist, max_age_hist, run_name)

    # show all plots
    plt.show()

    return pop, archive, hall_of_fame, details
