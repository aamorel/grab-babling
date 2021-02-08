import random
import numpy as np
from numpy import linalg as LA
from deap import tools, base
from sklearn.neighbors import NearestNeighbors as Nearest
from scipy.spatial import cKDTree as KDTree
import scipy.stats as stats
from scoop import futures
import utils
import math
from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import plotting
import tqdm

creator = None


def set_creator(cr):
    global creator
    creator = cr


HOF_SIZE = 10  # number of individuals in hall of fame
K = 15  # number of nearest neighbours for novelty computation
INF = 1000000000  # for security against infinite distances in KDtree queries
ARCHIVE = 'random'  # random or novelty_based
ARCHIVE_PB = 0.2  # if ARCHIVE is random, probability to add individual to archive
# if ARCHIVE is novelty_based, proportion of individuals added per gen
ARCHIVE_DECREMENTAL_RATIO = 0.9  # if archive size is bigger than thresh, cut down archive by this ratio
CXPB = 0.2  # probability with which two individuals are crossed
MUTPB = 0.8  # probability for mutating an individual
SIGMA = 0.1  # std of the mutation of one gene
# CXPB and MUTPB should sum up to 1
TOURNSIZE = 10  # drives towards selective pressure
OFFSPRING_NB_COEFF = 0.5  # number of offsprings generated (coeff of pop length)
MAX_MUT_PROB = 0.5  # if algo is ns_rand_keep_diversity, probability for the less diverse gene to be mutated
N_CELLS = 20  # number of cells to try to generate in case of grid_density archive management
LIMIT_DENSITY_ITER = 100  # maximum number of iterations to find an individual in the cell not already chosen
# in case of grid_density_archive_management
N_COMP = 4  # number of GMM components in case of gmm sampling archive management
GIF_LENGHT = 30  # length of gif in seconds
GIF_TYPE = 'hist_color'  # 'full' or 'hist_color'
CM = cm.get_cmap('viridis', 100)

id_counter = 0  # each individual will have a unique id


def initialize_tool(initial_gen_size, mini, pop_size, parallelize, algo_type):
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
        if algo_type == 'ns_rand_multi_bd':
            # novelty must be a list, and selection is not used directly with DEAP
            creator.create('Novelty', list)
        else:
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
        query (List): behavioral descriptor of individual
        k_tree (KDTree or Nearest): tree in the behavior descriptor space

    Returns:
        float: average distance to the K nearest neighbours
    """
    if isinstance(k_tree, KDTree):
        # no used anymore but kept in case
        # find K nearest neighbours and distances
        neighbours_distances = k_tree.query(query, range(2, K + 2))[0]
        # beware: if K > number of points in tree, missing neighbors are associated with infinite distances
        # workaround:
        real_neighbours_distances = neighbours_distances[neighbours_distances < INF]
        # compute mean distance
        avg_distance = np.mean(real_neighbours_distances)
    if isinstance(k_tree, Nearest):
        n_samples = k_tree.n_samples_fit_
        query = np.array(query)
        if n_samples >= K + 1:
            neighbours_distances = k_tree.kneighbors(X=query.reshape(1, -1))[0][0][1:]
        else:
            neighbours_distances = k_tree.kneighbors(X=query.reshape(1, -1), n_neighbors=n_samples)[0][0][1:]
        avg_distance = np.mean(neighbours_distances)
    return avg_distance,


def compute_average_distance_array(query, k_tree):
    """Finds K nearest neighbours and distances

    Args:
        query (List): behavioral descriptor of individual
        k_tree (Nearest): tree in the behavior descriptor space

    Returns:
        float: average distance to the K nearest neighbours
    """
    n_samples = k_tree.n_samples_fit_
    query = np.array(query)
    if n_samples >= K + 1:
        neighbours_distances = k_tree.kneighbors(X=query)[0][:, 1:]
    else:
        neighbours_distances = k_tree.kneighbors(X=query, n_neighbors=n_samples)[0][:, 1:]
    avg_distances = np.mean(neighbours_distances, axis=1)
    avg_distance_tuples = [(avg_dist,) for avg_dist in avg_distances]
    return avg_distance_tuples


def assess_novelties(pop, archive, algo_type, bd_bounds, bd_indexes, bd_filters, novelty_metric,
                     altered=False, degree=None, info=None):
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

    if algo_type == 'ns_rand_multi_bd':

        # create the different trees with the reference pop
        b_descriptors = np.array(b_descriptors)
        bd_indexes = np.array(bd_indexes)
        nb_bd = len(np.unique(bd_indexes))
        k_trees = []  # will contain the kd trees for the different bds
        bd_lists = [[] for _ in range(nb_bd)]  # will contain the lists of different bds values
        for bd in b_descriptors:
            for idx, bd_filter in enumerate(bd_filters):
                bd_value = bd[bd_filter]
                if not(None in bd_value):  # if the bd has a value
                    bd_lists[idx].append(bd[bd_filter])
        for idx in range(nb_bd):
            if len(bd_lists[idx]) > 0:
                neigh = Nearest(n_neighbors=K + 1, metric=novelty_metric[idx])
                neigh.fit(bd_lists[idx])
                k_trees.append(neigh)
            else:
                k_trees.append(None)

        # compute novelties for the pop bds
        for i in range(len(pop)):
            # in that case, novelty will be a tupple of novelties for each bd
            bd = b_descriptors[i]

            # loop through the bds
            novelty = []
            for idx, bd_filter in enumerate(bd_filters):
                bd_value = bd[bd_filter]
                if not(None in bd_value):
                    nov_bd = compute_average_distance(bd_value, k_trees[idx])[0]  # float
                    novelty.append(nov_bd)
                else:
                    novelty.append(None)
            novelty = tuple(novelty)
            novelties.append(novelty)
            
    else:
        # extract all the behavior descriptors that are not None to create the tree
        b_ds = [ind.behavior_descriptor.values for ind in reference_pop if ind.behavior_descriptor.values is not None]
        k_tree = Nearest(n_neighbors=K + 1, metric=novelty_metric)
        k_tree.fit(b_ds)
        # compute novelty for current individuals (loop only on the pop)
        if algo_type == 'ns_rand_change_bd':
            # behavior descriptors can be None
            for i in range(len(pop)):
                if b_descriptors[i] is not None:
                    novelties.append(compute_average_distance(b_descriptors[i], k_tree))
                else:
                    novelties.append((0.0,))
        else:
            novelties = compute_average_distance_array(b_descriptors, k_tree)

        if altered:
            # experimental condition: alter the novelties

            # compute ranking before
            nov_n = np.array(novelties).flatten()
            order = nov_n.argsort()
            ranking_before = order.argsort()

            mean_nov = np.mean(nov_n)
            rand_range = mean_nov * degree
            for i, nov in enumerate(novelties):
                # each novelty is incremented by a random float between -rand_range and rang_range
                novelties[i] = (nov[0] + (random.random() * 2 - 0.5) * rand_range,)
            
            # compute ranking after (using altered novelties)
            nov_n = np.array(novelties).flatten()
            order = nov_n.argsort()
            ranking_after = order.argsort()

            # compute rankings similarity
            ranking_similarity = stats.kendalltau(ranking_before, ranking_after)[0]
            info['ranking similarities novelty'].append(ranking_similarity)

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
    std_genes = np.std(gen_stats, axis=1).tolist()
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


def add_to_grid(member, grid, cvt, measures, algo_type, bd_filters):
    if measures:
        member_bd = np.array(member.behavior_descriptor.values)
        if algo_type == 'ns_rand_multi_bd':
            # grid and cvt are a list of grids and cvts
            for idx, bd_filter in enumerate(bd_filters):
                bd_value = member_bd[bd_filter]
                if not(None in bd_value):  # if the bd has a value
                    grid_index = cvt[idx].get_grid_index(bd_value)
                    grid[idx][grid_index] += 1
        else:
            grid_index = cvt.get_grid_index(member_bd)
            grid[grid_index] += 1


def remove_from_grid(member, grid, cvt, measures, algo_type, bd_filters):
    if measures:
        member_bd = np.array(member.behavior_descriptor.values)
        if algo_type == 'ns_rand_multi_bd':
            # grid and cvt are a list of grids and cvts
            for idx, bd_filter in enumerate(bd_filters):
                bd_value = member_bd[bd_filter]
                if not(None in bd_value):  # if the bd has a value
                    grid_index = cvt[idx].get_grid_index(bd_value)
                    grid[idx][grid_index] -= 1
        else:
            grid_index = cvt.get_grid_index(member_bd)
            grid[grid_index] -= 1


def select_n_multi_bd(pop, n, putback=True):
    selected = []
    unwanted_list = []  # in case of no putback
    pop_size = len(pop)
    for i in range(n):
        # make sure two selected individuals are different
        condition = True
        while condition:
            idx1 = random.randint(0, pop_size - 1)
            idx2 = random.randint(0, pop_size - 1)
            if putback:
                condition = idx1 == idx2
            else:
                condition = ((idx1 == idx2) or (idx1 in unwanted_list)) or (idx2 in unwanted_list)
        ind1_nov = list(pop[idx1].novelty.values)
        ind2_nov = list(pop[idx2].novelty.values)

        # find common bds between the two individuals
        common_bds = []
        for i in range(len(ind1_nov)):
            if (ind1_nov[i] is not None) and (ind2_nov[i] is not None):
                common_bds.append(i)
        nb_common_bds = len(common_bds)
        if nb_common_bds == 0:
            # no common bds
            # hypothesis: choose one of the two individuals randomly
            dice = random.randint(0, 1)
            if dice:
                selected.append(pop[idx1])
                unwanted_list.append(idx1)
            else:
                selected.append(pop[idx2])
                unwanted_list.append(idx2)
        else:
            # choose a random common bd
            common_bd_idx = random.randint(0, nb_common_bds - 1)
            bd_idx = common_bds[common_bd_idx]
            ind1_nov_val = ind1_nov[bd_idx]
            ind2_nov_val = ind2_nov[bd_idx]
            if ind1_nov_val >= ind2_nov_val:
                selected.append(pop[idx1])
                unwanted_list.append(idx1)
            else:
                selected.append(pop[idx2])
                unwanted_list.append(idx2)
            
    return selected


def select_n_multi_bd_tournsize(pop, n, tournsize, bd_filters, multi_quality, putback=True):
    # decide if should use quality or not
    use_quality = False
    if multi_quality is not None:
        use_quality = True

    selected = []
    unwanted_list = []  # in case of no putback
    pop_size = len(pop)
    nb_of_bd = len(bd_filters)
    for i in range(n):
        # prepare the tournament
        # make sure selected individuals are different
        if putback:
            tourn_idxs = random.sample(range(pop_size), tournsize)
        else:
            list_of_available_idxs = [i for i in list(range(pop_size)) if i not in unwanted_list]
            random.shuffle(list_of_available_idxs)
            tourn_idxs = list_of_available_idxs[:tournsize]

        # make the inventory of the bds in the tournament
        inventory = np.zeros(nb_of_bd)
        for idx in tourn_idxs:
            ind = pop[idx]
            nov_list = list(ind.novelty.values)
            for i, nov in enumerate(nov_list):
                if nov is not None:
                    inventory[i] += 1
        
        # choose the bd to use for comparison
        bd_idx = choose_bd_strategy(inventory)

        # find all the individuals that are evaluated inside the chosen bd and their novelties
        possible_individuals_idxs = []
        possible_individuals_novelties = []

        if use_quality:
            possible_individuals_qualities = []
            minimization = multi_quality[1][bd_idx] == 'min'

        for idx in tourn_idxs:
            ind = pop[idx]
            nov_list = list(ind.novelty.values)
            nov_to_compare = nov_list[bd_idx]
            if nov_to_compare is not None:
                possible_individuals_idxs.append(idx)
                possible_individuals_novelties.append(nov_to_compare)
                if use_quality:
                    name_of_quality = multi_quality[0][bd_idx]
                    quality_to_compare = ind.info.values[name_of_quality]
                    possible_individuals_qualities.append(quality_to_compare)
        
        # find most novel individual and select it
        # sanity check
        assert(len(possible_individuals_novelties) > 0)

        possible_individuals_novelties = np.array(possible_individuals_novelties)
        if not use_quality:
            # simply choose the individual maximizing the novelty
            temp_idx = np.argmax(possible_individuals_novelties)
        else:
            n_possible_inds = len(possible_individuals_novelties)
            possible_individuals_qualities = np.array(possible_individuals_qualities)
            novelties_order = possible_individuals_novelties.argsort()
            novelties_ranking = novelties_order.argsort()  # 0 is the least novel, n_possible_inds - 1 is the most novel

            qualities_order = possible_individuals_qualities.argsort()
            qualities_ranking_temp = qualities_order.argsort()
            if minimization:
                qualities_ranking = [n_possible_inds - 1 - rk for rk in qualities_ranking_temp]
                qualities_ranking = np.array(qualities_ranking)  # 0 is the least qualitative, n_possible_inds - 1 best
            final_rankings = novelties_ranking + qualities_ranking

            # choose the best one in terms of novelty ranking and quality ranking
            temp_idx = np.argmax(final_rankings)

        ind_idx = possible_individuals_idxs[temp_idx]
        selected.append(pop[ind_idx])
        if not putback:
            unwanted_list.append(ind_idx)
            
    return selected


def choose_bd_strategy(inventory):
    """Choose the behavior descriptor to use for comparison of novelties in the case of multi_bd novelty search and
    a tournament size >= 2.

    Args:
        inventory (np.array): length of nb_bd, each value counts the number of times the particular bd is evaluated
                              inside the tournament

    Returns:
        int: index of the chosen bd
    """

    # most basic strategy: choose a random behavior descriptor, but make sure inventory(bd_index) > 0
    cond = False
    while not cond:
        bd_idx = random.randint(0, len(inventory) - 1)
        if inventory[bd_idx] > 0:
            cond = True

    return bd_idx


def novelty_algo(evaluate_individual_list, initial_gen_size, bd_bounds_list, mini=True, plot=False, nb_gen=100,
                 algo_type='ns_nov', bound_genotype=1, pop_size=30, parallelize=False,
                 measures=False, choose_evaluate=None, bd_indexes=None, archive_limit_size=None,
                 archive_limit_strat='random', nb_cells=1000, analyze_archive=False, altered_novelty=False,
                 alteration_degree=None, novelty_metric='minkowski', save_ind_cond=None, plot_gif=False,
                 bootstrap_individuals=None, multi_quality=None):

    # initialize return dictionnaries
    details = {}
    figures = {}
    data = {}

    details['evaluation function'] = evaluate_individual_list.__name__
    details['genotype size'] = initial_gen_size
    details['bd bounds'] = bd_bounds_list
    details['minimization'] = mini
    details['nb of generations'] = nb_gen
    details['algo type'] = algo_type
    details['genotype bounds'] = bound_genotype
    details['pop size'] = pop_size
    details['choose evaluate'] = choose_evaluate
    details['bd indexes'] = bd_indexes
    details['archive limit size'] = archive_limit_size
    details['archive limit strat'] = archive_limit_strat
    details['nb of cells'] = nb_cells
    details['analyze archive'] = analyze_archive
    details['altered novelty'] = altered_novelty
    details['alteration degree'] = alteration_degree
    details['multi quality'] = multi_quality
    nov_metrics = []
    if isinstance(novelty_metric, list):
        for nov_met in novelty_metric:
            if isinstance(nov_met, str):
                nov_metrics.append(nov_met)
            else:
                nov_metrics.append(nov_met.__name__)
    else:
        if isinstance(novelty_metric, str):
            nov_metrics = novelty_metric
        else:
            nov_metrics = novelty_metric.__name__
    details['novelty metric'] = nov_metrics

    # each individual will have a unique id
    global id_counter

    creator, toolbox = initialize_tool(initial_gen_size, mini, pop_size, parallelize, algo_type)

    # initialize evaluate_individual function
    if algo_type == 'ns_rand_change_bd':
        evaluate_individual = evaluate_individual_list[0]
        bd_bounds = bd_bounds_list[0]
        eval_index = 0
        info_change_bd = {}
        info_change_bd['changed'] = False

    else:
        evaluate_individual = evaluate_individual_list
        bd_bounds = np.array(bd_bounds_list)

    # initialize population
    pop = toolbox.population()

    # bootstrap if necessary
    if bootstrap_individuals is not None:
        count = 0
        nb_to_boostrap = int(pop_size / 2)
        # boostrap on half the population
        cond = True
        while cond:
            # for each bootstrapping individual, change one individual from initial population
            for new_ind in bootstrap_individuals:
                if len(new_ind) != initial_gen_size:
                    print('Required genotypic size: ', initial_gen_size)
                    print('Given genotypic size: ', len(new_ind))
                    raise Exception('One of the boostrapping individuals does not have the required genotype length')
                else:
                    for j, gene in enumerate(new_ind):
                        pop[count][j] = gene
                    count += 1
                    if count >= nb_to_boostrap:
                        cond = False
                        break

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
    archive_nb = int(ARCHIVE_PB * pop_size * OFFSPRING_NB_COEFF)

    # initialize the save_ind list
    save_ind = []

    # initialize the HOF
    hall_of_fame = tools.HallOfFame(HOF_SIZE)

    # if altered novelty experimental condition
    if altered_novelty:
        details['ranking similarities novelty'] = []

    # grid contains the current content of the archive
    grid = None
    # grid_hist contains all individuals that have been created
    grid_hist = None
    # grid_pop contains the current content of the population
    grid_pop = None
    # cvt is the tool to attribute individuals to grid cells (used for both grid and grid_hist)
    cvt = None
    bd_filters = None

    if algo_type == 'ns_rand_multi_bd':
        bd_indexes = np.array(bd_indexes)
        nb_bd = len(np.unique(bd_indexes))

        if multi_quality is not None:
            if len(multi_quality[0]) != nb_bd:
                raise Exception('Number of quality measures not equal to number of behavioral descriptors.')

        bd_filters = []  # will contain the boolean filters for the different bds
        for idx in range(nb_bd):
            bd_filters.append(bd_indexes == idx)

    if measures:
        # initialize the CVT grid
        if algo_type == 'ns_rand_multi_bd':
            # grid and cvt will be lists for each BD
            grid = []
            grid_hist = []
            cvt = []
            for bd_filter in bd_filters:
                grid.append(np.zeros(nb_cells))
                grid_hist.append(np.zeros(nb_cells))
                cvt_member = utils.CVT(num_centroids=nb_cells, bounds=bd_bounds[bd_filter])
                cvt.append(cvt_member)
                
        else:
            grid = np.zeros(nb_cells)
            grid_hist = np.zeros(nb_cells)
            cvt = utils.CVT(num_centroids=nb_cells, bounds=bd_bounds)
    
    # initialize ranking similarities and novelty differences lists if analyzing the archive
    ranking_similarities = []
    novelty_differences = []

    # evaluate initial population
    evaluation_pop = list(toolbox.map(evaluate_individual, pop))
    b_descriptors, fitnesses, infos = map(list, zip(*evaluation_pop))

    # attribute fitness and behavior descriptors to individuals
    for ind, fit, bd, inf in zip(pop, fitnesses, b_descriptors, infos):
        ind.behavior_descriptor.values = bd
        ind.info.values = inf
        ind.fitness.values = fit

    novelties = assess_novelties(pop, archive, algo_type, bd_bounds, bd_indexes, bd_filters, novelty_metric,
                                 altered=altered_novelty, degree=alteration_degree, info=details)
    for ind, nov in zip(pop, novelties):
        ind.novelty.values = nov

    # add initial individuals to historic and potentially to saved individuals
    if save_ind_cond == 1:
        save_ind.append(list(map(toolbox.clone, pop)))
    for member in pop:
        add_to_grid(member, grid_hist, cvt, measures, algo_type, bd_filters)
        if save_ind_cond is None:
            pass
        if isinstance(save_ind_cond, str):
            if member.info.values[save_ind_cond]:
                save_ind.append(member)

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
        novelty_distrib = []
        coverage_hist = []
        full_cov_hist = []
        pop_cov_hist = []
        uniformity_hist = []
        full_uni_hist = []
        pop_uni_hist = []

    if plot_gif:
        fig_gif = plt.figure(figsize=(10, 10))
        ims = []

    # begin evolution
    for gen in tqdm.tqdm(range(nb_gen)):

        # ###################################### SELECT ############################################
        nb_offsprings_to_generate = int(pop_size * OFFSPRING_NB_COEFF)
        # references to selected individuals
        if algo_type == 'classic_ea':
            # classical EA: selection on fitness
            offsprings = toolbox.select(pop, nb_offsprings_to_generate, fit_attr='fitness')
        elif algo_type == 'ns_rand_multi_bd':
            # use special selection for multi novelties
            offsprings = select_n_multi_bd_tournsize(pop, nb_offsprings_to_generate, TOURNSIZE,
                                                     bd_filters, multi_quality)
        elif algo_type == 'random_search':
            # for experimental baseline
            offsprings = random.sample(pop, nb_offsprings_to_generate)
        else:
            # novelty search: selection on novelty
            offsprings = toolbox.select(pop, nb_offsprings_to_generate, fit_attr='novelty')
        
        offsprings = list(map(toolbox.clone, offsprings))  # clone selected indivduals

        # ###################################### MUTATE ############################################
        # for now, offsprings are clones of parents --> they keep the genetic information
        if algo_type == 'ns_rand_keep_diversity':
            operate_offsprings_diversity(offsprings, toolbox, bound_genotype,
                                         pop)  # crossover and mutation
        else:
            operate_offsprings(offsprings, toolbox, bound_genotype)  # crossover and mutation
        # now, offsprings have their correct genetic information

        # ###################################### EVALUATE ############################################
        # current pool is old population + generated offsprings
        current_pool = pop + offsprings

        # evaluate the individuals with an invalid fitness
        # done for the whole current pool and not just the offsprings in case the evaluation
        # function has changed, but in the general case only done for the offsprings
        invalid_ind = [ind for ind in current_pool if not ind.fitness.valid]
        evaluation_pop = list(toolbox.map(evaluate_individual, invalid_ind))
        inv_b_descriptors, inv_fitnesses, inv_infos = map(list, zip(*evaluation_pop))

        # attribute fitness and behavior descriptors to new individuals
        for ind, fit, bd, inf in zip(invalid_ind, inv_fitnesses, inv_b_descriptors, inv_infos):
            ind.behavior_descriptor.values = bd  # can be None in the change_bd case
            ind.info.values = inf
            ind.fitness.values = fit

        # compute novelty for all current individuals (novelty of population may have changed)
        novelties = assess_novelties(current_pool, archive, algo_type, bd_bounds, bd_indexes, bd_filters,
                                     novelty_metric,
                                     altered=altered_novelty, degree=alteration_degree, info=details)
        if measures:
            novelty_distrib.append(novelties)
        for ind, nov in zip(current_pool, novelties):
            ind.novelty.values = nov
        # an individual with bd = None will have 0 novelty

        # add all generated individuals to historic and potentially to saved individuals
        if save_ind_cond == 1:
            save_ind.append(offsprings)
        for member in offsprings:
            add_to_grid(member, grid_hist, cvt, measures, algo_type, bd_filters)
            if save_ind_cond is None:
                pass
            if isinstance(save_ind_cond, str):
                if member.info.values[save_ind_cond]:
                    save_ind.append(member)

        # ###################################### FILL ARCHIVE ############################################
        # fill archive with individuals from the offsprings group (direct references to those individuals)
        # grid follows the archive
        if algo_type != 'ns_no_archive':
            if archive_type == 'random':
                # fill archive randomly
                fill_arch_count = 0
                idx_list = []
                while fill_arch_count < archive_nb:
                    idx = random.randint(0, len(offsprings) - 1)
                    if idx not in idx_list:
                        member = offsprings[idx]
                        archive.append(member)
                        add_to_grid(member, grid, cvt, measures, algo_type, bd_filters)
                        idx_list.append(idx)
                        fill_arch_count += 1

            if archive_type == 'novelty_based':
                #  TODO: deal with multi-bd case
                # fill archive with the most novel individuals
                offsprings_novelties = novelties[pop_size:]
                novel_n = np.array([nov[0] for nov in offsprings_novelties])
                max_novelties_idx = np.argsort(-novel_n)[:archive_nb]
                for i in max_novelties_idx:
                    if offsprings[i].behavior_descriptor.values is not None:
                        member = offsprings[i]
                        archive.append(member)
                        add_to_grid(member, grid, cvt, measures, algo_type, bd_filters)
        
        # ###################################### REPLACE ############################################
        if algo_type == 'classic_ea':
            # replacement: keep the most fit individuals
            pop[:] = toolbox.replace(current_pool, pop_size, fit_attr='fitness')
        elif algo_type == 'ns_rand_multi_bd':
            # replacement: keep the most novel individuals in case of multi novelties
            pop[:] = select_n_multi_bd_tournsize(current_pool, pop_size, TOURNSIZE,
                                                 bd_filters, multi_quality, putback=False)
        elif algo_type == 'random_search':
            pop[:] = random.sample(current_pool, pop_size)
        else:
            # replacement: keep the most novel individuals
            pop[:] = toolbox.replace(current_pool, pop_size, fit_attr='novelty')

        if algo_type == 'ns_rand_binary_removal':
            # remove individuals that satisfy the binary goal
            # they can still be in the archive
            for i, ind in enumerate(pop):
                if ind.info.values['binary goal']:
                    pop.pop(i)

        # ###################################### MANAGE ARCHIVE ############################################
        if archive_limit_size is not None:
            # implement archive size limitation strategy
            if len(archive) >= archive_limit_size:

                if analyze_archive:
                    # monitor the change of ranking of novelties of population
                    novelties = assess_novelties(pop, pop + archive, algo_type, bd_bounds, bd_indexes, bd_filters,
                                                 novelty_metric)
                    nov_n_before = np.array([nov[0] for nov in novelties])
                    order = nov_n_before.argsort()
                    ranking_before = order.argsort()

                original_len = len(archive)
                nb_ind_to_keep = int(original_len * ARCHIVE_DECREMENTAL_RATIO)
                nb_ind_to_remove = original_len - nb_ind_to_keep

                # removal strategies
                if archive_limit_strat == 'random':
                    # strategy 1: remove random individuals
                    random.shuffle(archive)
                    # remove from grid
                    members_to_remove = archive[nb_ind_to_keep:]
                    for member in members_to_remove:
                        remove_from_grid(member, grid, cvt, measures, algo_type, bd_filters)

                    archive = archive[:nb_ind_to_keep]
                
                if archive_limit_strat == 'oldest':
                    # strategy 2: remove oldest individuals
                    members_to_remove = archive[:nb_ind_to_remove]
                    for member in members_to_remove:
                        remove_from_grid(member, grid, cvt, measures, algo_type, bd_filters)
                    archive = archive[nb_ind_to_remove:]

                if archive_limit_strat == 'least_novel':
                    # strategy 3: remove least novel individuals
                    novelties = assess_novelties(archive, archive, algo_type, bd_bounds, bd_indexes, bd_filters,
                                                 novelty_metric)
                    nov_n = np.array([nov[0] for nov in novelties])
                    removal_indices = np.argpartition(nov_n, nb_ind_to_remove)[:nb_ind_to_remove]
                    for idx in removal_indices:
                        remove_from_grid(archive[idx], grid, cvt, measures, algo_type, bd_filters)
                    temp_archive = []
                    for i in range(original_len):
                        if i not in removal_indices:
                            temp_archive.append(archive[i])
                    archive = temp_archive

                if archive_limit_strat == 'least_novel_iter':
                    # strategy 4: remove least novel individual iteratively
                    for _ in range(nb_ind_to_remove):
                        novelties = assess_novelties(archive, archive, algo_type, bd_bounds, bd_indexes, bd_filters,
                                                     novelty_metric)
                        nov_n = np.array([nov[0] for nov in novelties])
                        removal_idx = np.argmin(nov_n)
                        remove_from_grid(archive[removal_idx], grid, cvt, measures, algo_type, bd_filters)
                        archive.pop(removal_idx)
                
                if archive_limit_strat == 'grid_density':
                    # strategy 5: remove individuals with probability proportional to grid density
                    # extract all the behavior descriptors
                    reference_pop = np.array([ind.behavior_descriptor.values for ind in archive])
                    n_dim = reference_pop.shape[1]
                    # compute maximums and mins on each dimension
                    maximums = np.max(reference_pop, 0)
                    mins = np.min(reference_pop, 0)
                    ranges = maximums - mins
                    bins_per_dim = math.floor(math.exp(math.log(N_CELLS) / n_dim)) + 1
                    grid_positions = []
                    for i in range(n_dim):
                        # important choice on how we make the grid
                        grid_position = [mins[i] + (j * ranges[i] / (bins_per_dim - 1)) for j in range(bins_per_dim)]
                        grid_positions.append(grid_position)
                    mesh = np.meshgrid(*grid_positions)
                    nodes = list(zip(*(dim.flat for dim in mesh)))
                    nodes = np.array(nodes)

                    removal_indices = []
                    n_cells = (bins_per_dim - 1) ** n_dim
                    grid_density = np.zeros(n_cells)
                    cells = [[] for _ in range(n_cells)]

                    for ind_idx, ind in enumerate(reference_pop):
                        dim_indexs = np.zeros(n_dim)
                        for i, dim in enumerate(ind):
                            grid_pos = grid_positions[i]
                            for j in range(bins_per_dim - 1):
                                if dim >= grid_pos[j] and dim < grid_pos[j + 1]:
                                    dim_indexs[i] = j + 1
                        if 0 not in dim_indexs:
                            # indivudal is inside the grid
                            dim_indexs = dim_indexs - 1
                            cell_idx = 0
                            for k, dim_idx in enumerate(dim_indexs):
                                cell_idx += int(dim_idx * ((bins_per_dim - 1) ** k))
                            grid_density[cell_idx] += 1
                            cells[cell_idx].append(ind_idx)
                    
                    grid_density = grid_density / np.sum(grid_density)

                    grid_law = np.cumsum(grid_density)

                    for _ in range(nb_ind_to_remove):
                        dice = random.random() * grid_law[-1]
                        cell_to_remove_from = np.searchsorted(grid_law, dice)
                        cond = True
                        n = 0
                        while cond:
                            if n < LIMIT_DENSITY_ITER:
                                removal_idx = random.choice(cells[cell_to_remove_from])
                            else:
                                removal_idx = random.choice(list(range(len(reference_pop))))
                            if removal_idx not in removal_indices:
                                removal_indices.append(removal_idx)
                                cond = False
                            n += 1
                    for idx in removal_indices:
                        remove_from_grid(archive[idx], grid, cvt, measures, algo_type, bd_filters)
                    temp_archive = []
                    for i in range(original_len):
                        if i not in removal_indices:
                            temp_archive.append(archive[i])
                    archive = temp_archive
                    
                if archive_limit_strat == 'grid':
                    # strategy 6: remove individuals at intersection of grid
                    # extract all the behavior descriptors
                    reference_pop = np.array([ind.behavior_descriptor.values for ind in archive])
                    n_dim = reference_pop.shape[1]
                    # compute maximums and mins on each dimension
                    maximums = np.max(reference_pop, 0)
                    mins = np.min(reference_pop, 0)
                    ranges = maximums - mins
                    bins_per_dim = math.floor(math.exp(math.log(nb_ind_to_remove) / n_dim)) + 1
                    grid_positions = []
                    for i in range(n_dim):
                        # important choice on how we make the grid
                        grid_position = [mins[i] + ((j + 1) * ranges[i] / bins_per_dim) for j in range(bins_per_dim)]
                        grid_position.pop()
                        grid_positions.append(grid_position)
                    mesh = np.meshgrid(*grid_positions)
                    nodes = list(zip(*(dim.flat for dim in mesh)))
                    nodes = np.array(nodes)

                    k_tree = KDTree(reference_pop)
                    removal_indices = []
                    for node in nodes:
                        # for each node, find the closest point in the reference pop
                        cond = True
                        closest = 1
                        # make sure removal indivual was not already chosen
                        while cond:
                            if closest == 1:
                                possible_removal_index = k_tree.query(node, closest)[1]
                            else:
                                possible_removal_index = k_tree.query(node, closest)[1][closest - 1]
                            if possible_removal_index not in removal_indices:
                                removal_indices.append(possible_removal_index)
                                cond = False
                            else:
                                closest += 1
                    # dealing with the missing removals
                    nb_missing_removals = nb_ind_to_remove - len(nodes)
                    for _ in range(nb_missing_removals):
                        query = random.choice(nodes)
                        cond = True
                        # start with second closest since closest is for sure in removal indices
                        closest = 2
                        # make sure removal indivual was not already chosen
                        while cond:
                            possible_removal_index = k_tree.query(query, closest)[1][closest - 1]
                            if possible_removal_index not in removal_indices:
                                removal_indices.append(possible_removal_index)
                                cond = False
                            else:
                                closest += 1
                    for idx in removal_indices:
                        remove_from_grid(archive[idx], grid, cvt, measures, algo_type, bd_filters)
                    temp_archive = []
                    for i in range(original_len):
                        if i not in removal_indices:
                            temp_archive.append(archive[i])
                    archive = temp_archive

                if archive_limit_strat == 'gmm':
                    # strategy 7: fit a gmm on archive, sample and remove closest
                    # extract all the behavior descriptors
                    reference_pop = np.array([ind.behavior_descriptor.values for ind in archive])
                    gmix = mixture.GaussianMixture(n_components=N_COMP, covariance_type='full')
                    gmix.fit(reference_pop)
                    nodes = gmix.sample(nb_ind_to_remove)[0]
                    k_tree = KDTree(reference_pop)
                    removal_indices = []
                    for node in nodes:
                        # for each node, find the closest point in the reference pop
                        cond = True
                        closest = 1
                        # make sure removal indivual was not already chosen
                        while cond:
                            if closest == 1:
                                possible_removal_index = k_tree.query(node, closest)[1]
                            else:
                                possible_removal_index = k_tree.query(node, closest)[1][closest - 1]
                            if possible_removal_index not in removal_indices:
                                removal_indices.append(possible_removal_index)
                                cond = False
                            else:
                                closest += 1
                    for idx in removal_indices:
                        remove_from_grid(archive[idx], grid, cvt, measures, algo_type, bd_filters)
                    temp_archive = []
                    for i in range(original_len):
                        if i not in removal_indices:
                            temp_archive.append(archive[i])
                    archive = temp_archive

                if archive_limit_strat == 'newest':
                    # strategy 8: remove newest individuals (blocks the archive sanity check)
                    members_to_remove = archive[nb_ind_to_keep:]
                    for member in members_to_remove:
                        remove_from_grid(member, grid, cvt, measures, algo_type, bd_filters)
                    archive = archive[:nb_ind_to_keep]
                
                if archive_limit_strat == 'most_novel':
                    # strategy 9: remove most novel individuals
                    novelties = assess_novelties(archive, archive, algo_type, bd_bounds, bd_indexes, bd_filters,
                                                 novelty_metric)
                    nov_n = np.array([nov[0] for nov in novelties])
                    removal_indices = np.argpartition(nov_n, -nb_ind_to_remove)[-nb_ind_to_remove:]
                    for idx in removal_indices:
                        remove_from_grid(archive[idx], grid, cvt, measures, algo_type, bd_filters)
                    temp_archive = []
                    for i in range(original_len):
                        if i not in removal_indices:
                            temp_archive.append(archive[i])
                    archive = temp_archive

                if archive_limit_strat == 'most_novel_iter':
                    # strategy 10: remove most novel individual iteratively
                    for _ in range(nb_ind_to_remove):
                        novelties = assess_novelties(archive, archive, algo_type, bd_bounds, bd_indexes, bd_filters,
                                                     novelty_metric)
                        nov_n = np.array([nov[0] for nov in novelties])
                        removal_idx = np.argmax(nov_n)
                        remove_from_grid(archive[removal_idx], grid, cvt, measures, algo_type, bd_filters)
                        archive.pop(removal_idx)

                assert((original_len - len(archive)) == nb_ind_to_remove)
                if analyze_archive:
                    # monitor the change of ranking of novelties of population
                    novelties = assess_novelties(pop, pop + archive, algo_type, bd_bounds, bd_indexes, bd_filters,
                                                 novelty_metric)
                    nov_n_after = np.array([nov[0] for nov in novelties])
                    order = nov_n_after.argsort()
                    ranking_after = order.argsort()
                    ranking_similarity = stats.kendalltau(ranking_before, ranking_after)[0]
                    ranking_similarities.append(ranking_similarity)
                    novelty_mean_difference = LA.norm(nov_n_before - nov_n_after)
                    novelty_differences.append(novelty_mean_difference)

        # ###################################### MEASURE ############################################
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
            # re-build population grid (new grid at each generation)
            if algo_type == 'ns_rand_multi_bd':
                grid_pop = []
                for _ in range(nb_bd):
                    grid_pop.append(np.zeros(nb_cells))
            else:
                grid_pop = np.zeros(nb_cells)

            # populate the population grid
            for ind in pop:
                add_to_grid(ind, grid_pop, cvt, measures, algo_type, bd_filters)

            if algo_type == 'ns_rand_multi_bd':
                coverages = []
                uniformities = []
                # loop through all grids and compute measures for each grid
                for gr in grid:
                    coverages.append(np.count_nonzero(gr) / nb_cells)
                    uniformities.append(utils.compute_uniformity(gr))
                uniformity_hist.append(uniformities)
                coverage_hist.append(coverages)

                coverages = []
                uniformities = []
                # loop through all grids and compute measures for each grid
                for gr in grid_hist:
                    coverages.append(np.count_nonzero(gr) / nb_cells)
                    uniformities.append(utils.compute_uniformity(gr))
                full_uni_hist.append(uniformities)
                full_cov_hist.append(coverages)

                coverages = []
                uniformities = []
                # loop through all grids and compute measures for each grid
                for gr in grid_pop:
                    coverages.append(np.count_nonzero(gr) / nb_cells)
                    uniformities.append(utils.compute_uniformity(gr))
                pop_uni_hist.append(uniformities)
                pop_cov_hist.append(coverages)

            else:
 
                # compute coverage
                coverage = np.count_nonzero(grid) / nb_cells
                coverage_hist.append(coverage)
                coverage = np.count_nonzero(grid_hist) / nb_cells
                full_cov_hist.append(coverage)
                coverage = np.count_nonzero(grid_pop) / nb_cells
                pop_cov_hist.append(coverage)

                # compute uniformity
                uniformity = utils.compute_uniformity(grid)
                uniformity_hist.append(uniformity)
                uniformity = utils.compute_uniformity(grid_hist)
                full_uni_hist.append(uniformity)
                uniformity = utils.compute_uniformity(grid_pop)
                pop_uni_hist.append(uniformity)

        arch_size_hist.append(len(archive))
        mean_hist.append(mean_fit)
        min_hist.append(np.min(fits))
        max_hist.append(np.max(fits))
        max_age_hist.append(int(np.max(ages)))
        mean_age_hist.append(mean_age)

        if algo_type == 'ns_rand_change_bd':
            # novelty search expects a list of evaluation functions and a list of bd_bounds
            # and a choose_evaluate functions which chooses which evaluation function to use
            # based on the info_change_bd dictionnary (returns an index)
            info_change_bd['coverage'] = coverage
            
            old_eval_index = eval_index
            eval_index = choose_evaluate(info_change_bd)
            evaluate_individual = evaluate_individual_list[eval_index]
            bd_bounds = bd_bounds_list[eval_index]
            if old_eval_index != eval_index:
                # behavior descriptor has changed
                # add info
                info_change_bd['changed'] = True
                
                # empty the archive
                archive = []

                # set the current population to be re-evaluated
                for ind in pop:
                    del ind.fitness.values

                if measures:
                    # re-initialize the CVT grid
                    grid = np.zeros(nb_cells)
                    cvt = utils.CVT(num_centroids=nb_cells, bounds=bd_bounds)
        
        if plot_gif:
            if GIF_TYPE == 'full':
                im_l = []
                if save_ind_cond == 1:
                    bds = []
                    for i, layers in enumerate(save_ind):
                        for member in layers:
                            bds.append(member.behavior_descriptor.values)
                    bds_arr = np.array(bds)

                    im_l.append(plt.scatter(bds_arr[:, 0], bds_arr[:, 1], color='grey', label='Historic'))
                archive_b = np.array([ind.behavior_descriptor.values for ind in archive])
                if len(archive) > 0:
                    im_l.append(plt.scatter(archive_b[:, 0], archive_b[:, 1], color='red', label='Archive'))
                pop_behavior = np.array([ind.behavior_descriptor.values for ind in pop])
                hof_behavior = np.array([ind.behavior_descriptor.values for ind in hall_of_fame])
                im_l.append(plt.scatter(pop_behavior[:, 0], pop_behavior[:, 1], color='blue', label='Population'))
                im_l.append(plt.scatter(hof_behavior[:, 0], hof_behavior[:, 1], color='green', label='Hall of Fame'))
                ims.append(im_l)
            if GIF_TYPE == 'hist_color':
                im_l = []
                for i, layers in enumerate(save_ind):
                    bds = []
                    for member in layers:
                        bds.append(member.behavior_descriptor.values)

                    bds_arr = np.array(bds)
                    color = CM(i / nb_gen)
                    im_l.append(plt.scatter(bds_arr[:, 0], bds_arr[:, 1], color=color, label='Historic'))
                ims.append(im_l)
    data['population genetic statistics'] = gen_stat_hist
    data['offsprings genetic statistics'] = gen_stat_hist_off
    data['archive coverage'] = coverage_hist
    data['archive uniformity'] = uniformity_hist
    data['coverage'] = full_cov_hist
    data['uniformity'] = full_uni_hist
    data['ranking similarities'] = ranking_similarities
    data['novelty differences'] = novelty_differences
    data['mean fitness'] = mean_hist
    data['min fitness'] = min_hist
    data['max fitness'] = max_hist
    data['archive size'] = arch_size_hist
    data['mean age'] = mean_age_hist
    data['max_age_hist'] = max_age_hist
    data['population coverage'] = pop_cov_hist
    data['population uniformity'] = pop_uni_hist
    data['novelty distribution'] = novelty_distrib

    if plot:
        fig, fig_2 = plotting.plot_launch(details, data)
    
        figures['figure'] = fig
        figures['figure_2'] = fig_2
    
    if plot_gif:
        interval = int(GIF_LENGHT * 1000 / len(ims))
        if interval >= 1000:
            interval = 1000
        if interval <= 50:
            interval = 50
        ani = animation.ArtistAnimation(fig_gif, ims, interval=interval, blit=False,
                                        repeat_delay=1000)
        figures['gif'] = ani

    return [pop, archive, hall_of_fame, details, figures, data, save_ind]
