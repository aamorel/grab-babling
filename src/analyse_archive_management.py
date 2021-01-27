import numpy as np
from scipy.spatial import cKDTree as KDTree
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
from sklearn import mixture
import time
import copy
import utils
import tqdm
from numpy import linalg as LA

K = 15
INF = 1000000000  # for security against infinite distances in KDtree queries
N = 3  # number of gaussians
CREATE_POP = 'gmm_based'  # type of population generation
SIGMA = 0.05  # mutation standard deviation for novelty_based population generation
LIMIT_DENSITY_ITER = 100  # maximum number of iterations to find an individual in the cell not already chosen
N_CELLS = 20  # number of cells to try to generate


def assess_novelties(reference_pop, pop):

    # empty list for novelties
    novelties = []
 
    k_tree = KDTree(np.concatenate((pop, reference_pop), 0))
    # compute novelty for current individuals (loop only on the pop)
    for p in pop:
        novelties.append(compute_average_distance(p, k_tree))
    novelties = np.array(novelties)
    return novelties


def compute_average_distance(query, k_tree):
    """Finds K nearest neighbours and distances

    Args:
        query (List): behavioral descriptor of individual
        k_tree (KDTree): tree in the behavior descriptor space

    Returns:
        float: average distance to the K nearest neighbours
    """
    # find K nearest neighbours and distances
    neighbours = k_tree.query(query, range(2, K + 2))[0]
    # beware: if K > number of points in tree, missing neighbors are associated with infinite distances
    # workaround:
    real_neighbours = neighbours[neighbours < INF]
    # compute mean distance
    avg_distance = np.mean(real_neighbours)
    return avg_distance


def compute_ranking(ref_pop, pop):
    novelties = assess_novelties(np.concatenate((ref_pop, pop), 0), pop)
    order = novelties.argsort()
    ranking = order.argsort()

    return ranking, novelties


def remove_ind(reference_pop, removal_size, removal_type):
    begin_time = time.time()

    if removal_type == 'random':
        # reference_pop is a numpy array of size (n_reference_pop, pop_dim)
        reference_pop = list(reference_pop)
        # now reference_pop is a list of numpy arrays (each defining one individual)
        random.shuffle(reference_pop)  # shuffle the list
        # pop last removal_size individuals
        for _ in range(removal_size):
            reference_pop.pop()
        # turn back to numpy array
        reference_pop = np.array(reference_pop)

    if removal_type == 'least_novel':
        # compute novelties of reference_pop inside reference_pop
        novelties = assess_novelties(reference_pop, reference_pop)
        removal_indices = np.argpartition(novelties, removal_size)[:removal_size]

        # # plot the reference pop
        # fig = plt.figure(figsize=(5, 5))
        # ax = fig.add_subplot(111)
        # ax.scatter(reference_pop[:, 0], reference_pop[:, 1], label='reference')
        # ax.scatter(reference_pop[removal_indices, 0], reference_pop[removal_indices, 1], label='removed',
        #            marker='x', color='red')
        # ax.set_facecolor("#ffebb8")
        # ax.set_title('Least novel removal', fontsize=15)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        # plt.legend()
        # plt.show()

        reference_pop = np.delete(reference_pop, removal_indices, 0)
    
    if removal_type == 'least_novel_iter':
        removal_indices = []
        temp_ref_pop = copy.deepcopy(reference_pop)
        for j in range(removal_size):
            # compute novelties of reference_pop inside reference_pop
            novelties = assess_novelties(temp_ref_pop, temp_ref_pop)
            remov_idx = np.argmin(novelties)
            remov_ind = temp_ref_pop[remov_idx]
            removal_indices.append(np.where(reference_pop == remov_ind)[0][0])
            temp_ref_pop = np.vstack((temp_ref_pop[:remov_idx], temp_ref_pop[remov_idx + 1:]))

        # # plot the reference pop
        # fig = plt.figure(figsize=(5, 5))
        # ax = fig.add_subplot(111)
        # ax.scatter(reference_pop[:, 0], reference_pop[:, 1], label='reference')
        # ax.scatter(reference_pop[removal_indices, 0], reference_pop[removal_indices, 1], label='removed',
        #            marker='x', color='red')
        # ax.set_facecolor("#ffebb8")
        # ax.set_title('Least novel removal', fontsize=15)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        # plt.legend()
        # plt.show()

        reference_pop = np.delete(reference_pop, removal_indices, 0)

    if removal_type == 'most_novel':
        # compute novelties of reference_pop inside reference_pop
        novelties = assess_novelties(reference_pop, reference_pop)
        removal_indices = np.argpartition(novelties, -removal_size)[-removal_size:]

        # # plot the reference pop
        # fig = plt.figure(figsize=(5, 5))
        # ax = fig.add_subplot(111)
        # ax.scatter(reference_pop[:, 0], reference_pop[:, 1], label='reference')
        # ax.scatter(reference_pop[removal_indices, 0], reference_pop[removal_indices, 1], label='removed',
        #            marker='x', color='red')
        # ax.set_facecolor("#ffebb8")
        # ax.set_title('Least novel removal', fontsize=15)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        # plt.legend()
        # plt.show()

        reference_pop = np.delete(reference_pop, removal_indices, 0)
    
    if removal_type == 'most_novel_iter':
        removal_indices = []
        temp_ref_pop = copy.deepcopy(reference_pop)
        for j in range(removal_size):
            # compute novelties of reference_pop inside reference_pop
            novelties = assess_novelties(temp_ref_pop, temp_ref_pop)
            remov_idx = np.argmax(novelties)
            remov_ind = temp_ref_pop[remov_idx]
            removal_indices.append(np.where(reference_pop == remov_ind)[0][0])
            temp_ref_pop = np.vstack((temp_ref_pop[:remov_idx], temp_ref_pop[remov_idx + 1:]))

        # # plot the reference pop
        # fig = plt.figure(figsize=(5, 5))
        # ax = fig.add_subplot(111)
        # ax.scatter(reference_pop[:, 0], reference_pop[:, 1], label='reference')
        # ax.scatter(reference_pop[removal_indices, 0], reference_pop[removal_indices, 1], label='removed',
        #            marker='x', color='red')
        # ax.set_facecolor("#ffebb8")
        # ax.set_title('Least novel removal', fontsize=15)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        # plt.legend()
        # plt.show()

        reference_pop = np.delete(reference_pop, removal_indices, 0)
    
    if removal_type == 'gmm_sampling':
        # hypothesis: n_components equals generative number of components
        n_comp = N
        gmix = mixture.GaussianMixture(n_components=n_comp, covariance_type='full')
        gmix.fit(reference_pop)
        nodes = gmix.sample(removal_size)[0]
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

        # # plot the reference pop
        # fig = plt.figure(figsize=(5, 5))
        # ax = fig.add_subplot(111)
        # ax.scatter(reference_pop[:, 0], reference_pop[:, 1], label='reference')
        # ax.scatter(reference_pop[removal_indices, 0], reference_pop[removal_indices, 1], label='removed',
        #            marker='x', color='red')
        # ax.set_facecolor("#ffebb8")
        # ax.set_title('GMM removal', fontsize=15)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        # plt.legend()
        # plt.show()

        reference_pop = np.delete(reference_pop, removal_indices, 0)

    if removal_type == 'grid':
        n_dim = reference_pop.shape[1]
        # compute maximums and minimums on each dimension
        maximums = np.max(reference_pop, 0)
        minimums = np.min(reference_pop, 0)
        ranges = maximums - minimums
        bins_per_dim = math.floor(math.exp(math.log(removal_size) / n_dim)) + 1
        grid_positions = []
        for i in range(n_dim):
            # important choice on how we make the grid
            grid_position = [minimums[i] + ((j + 1) * ranges[i] / bins_per_dim) for j in range(bins_per_dim)]
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
        nb_missing_removals = removal_size - len(nodes)
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
        
        # # plot the reference pop
        # fig = plt.figure(figsize=(5, 5))
        # ax = fig.add_subplot(111)
        # ax.scatter(reference_pop[:, 0], reference_pop[:, 1], label='reference')
        # ax.scatter(nodes[:, 0], nodes[:, 1], label='grid', marker='+', color='black')
        # ax.scatter(reference_pop[removal_indices, 0], reference_pop[removal_indices, 1], label='removed',
        #            marker='x', color='red')
        # ax.set_facecolor("#ffebb8")
        # ax.set_title('Grid removal', fontsize=15)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        # plt.legend()
        # plt.show()

        reference_pop = np.delete(reference_pop, removal_indices, 0)
    
    if removal_type == 'grid_density':
        n_dim = reference_pop.shape[1]
        # compute maximums and minimums on each dimension
        maximums = np.max(reference_pop, 0)
        minimums = np.min(reference_pop, 0)
        ranges = maximums - minimums
        bins_per_dim = math.floor(math.exp(math.log(N_CELLS) / n_dim)) + 1
        grid_positions = []
        for i in range(n_dim):
            # important choice on how we make the grid
            grid_position = [minimums[i] + (j * ranges[i] / (bins_per_dim - 1)) for j in range(bins_per_dim)]
            grid_positions.append(grid_position)
        mesh = np.meshgrid(*grid_positions)
        nodes = list(zip(*(dim.flat for dim in mesh)))
        nodes = np.array(nodes)

        removal_indices = []
        nb_cells = (bins_per_dim - 1) ** n_dim
        grid_density = np.zeros(nb_cells)
        cells = [[] for _ in range(nb_cells)]

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

        # TEST: square the grid_density to biase more towards high density cells
        # grid_density = np.square(grid_density)

        grid_law = np.cumsum(grid_density)

        for _ in range(removal_size):
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

        # # plot the reference pop
        # fig = plt.figure(figsize=(5, 5))
        # ax = fig.add_subplot(111)
        # ax.scatter(reference_pop[:, 0], reference_pop[:, 1], label='reference')
        # ax.scatter(nodes[:, 0], nodes[:, 1], label='grid', marker='+', color='black')
        # ax.scatter(reference_pop[removal_indices, 0], reference_pop[removal_indices, 1], label='removed',
        #            marker='x', color='red')
        # ax.set_facecolor("#ffebb8")
        # ax.set_title('Grid density removal', fontsize=15)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        # plt.legend()
        # plt.show()

        reference_pop = np.delete(reference_pop, removal_indices, 0)

    end_time = time.time()
    removal_time = end_time - begin_time
    return reference_pop, removal_time

    
def compare_rankings(rank_1, rank_2):
    return stats.kendalltau(rank_1, rank_2)


def create_pop(ref_pop_size, pop_size, pop_dim, generation_type):
    # all dimensions must stay in [0, 1]
    if generation_type == 'uniform':
        ref_pop = np.random.rand(ref_pop_size, pop_dim)
        pop = np.random.rand(pop_size, pop_dim)

    if generation_type == 'mixture':
        # create N means for gaussian model
        means = np.random.rand(N, pop_dim)
        covariances = np.random.rand(N, pop_dim) / 30
        weights = np.random.rand(N)
        weights = weights / np.sum(weights)
        gmix = mixture.GaussianMixture(n_components=N, covariance_type='diag')
        gmix.fit(np.random.rand(10, pop_dim))  # now it thinks it is trained
        gmix.weights_ = weights   # mixture weights (n_components,)
        gmix.means_ = means  # mixture means (n_components, pop_dim)
        gmix.covariances_ = covariances  # mixture cov (n_components, pop_dim, pop_dim)
        ref_pop = []
        while len(ref_pop) < ref_pop_size:
            x = gmix.sample()[0][0]
            cond = True
            for dim in x:
                if dim < 0 or dim > 1:
                    cond = False
            if cond:
                ref_pop.append(x)
        ref_pop = np.array(ref_pop)
        
        # choose how to create the population
        if CREATE_POP == 'gmm_based':
            pop = []
            while len(pop) < pop_size:
                x = gmix.sample()[0][0]
                cond = True
                for dim in x:
                    if dim < 0 or dim > 1:
                        cond = False
                if cond:
                    pop.append(x)
            pop = np.array(pop)

        if CREATE_POP == 'uniform':
            pop = np.random.rand(pop_size, pop_dim)

        if CREATE_POP == 'novelty_based':
            # idea: mimic a novelty search generation of new individuals
            pop = []

            # compute novelties of reference population with respect to itself
            novelties = assess_novelties(ref_pop, ref_pop)

            # find indices of the most novel individuals
            indices = (-novelties).argsort()[:pop_size]

            # create each pop individual by mutating one novel individual from ref_pop
            for ref_ind in ref_pop[indices]:
                cond = False
                while not cond:
                    # mutate each individual and make sure that the child stays in the bounds
                    pop_ind = ref_ind + np.random.normal(0, SIGMA, pop_dim)
                    cond = True
                    for dim in pop_ind:
                        if dim < 0 or dim > 1:
                            cond = False

                pop.append(pop_ind)
            pop = np.array(pop)
        
        if CREATE_POP == 'gmm_novelty':
            # idea: better mimic a novelty search generation of new individuals
            # sample the GMM for 2 * n points + mutate the most novel with respect to ref_pop
            pop_temp = []
            while len(pop_temp) < (2 * pop_size):
                x = gmix.sample()[0][0]
                cond = True
                for dim in x:
                    if dim < 0 or dim > 1:
                        cond = False
                if cond:
                    pop_temp.append(x)
            pop_temp = np.array(pop_temp)
            
            # compute novelties of reference population with respect to itself
            novelties = assess_novelties(ref_pop, pop_temp)

            # find indices of the most novel individuals
            indices = (-novelties).argsort()[:pop_size]

            pop_temp = pop_temp[indices]

            pop = []
            # create each pop individual by mutating one novel individual generated GMM samples
            for ref_ind in pop_temp:
                cond = False
                while not cond:
                    # mutate each individual and make sure that the child stays in the bounds
                    pop_ind = ref_ind + np.random.normal(0, SIGMA, pop_dim)
                    cond = True
                    for dim in pop_ind:
                        if dim < 0 or dim > 1:
                            cond = False

                pop.append(pop_ind)
            pop = np.array(pop)

    return ref_pop, pop


def analyze(ref_pop_size, pop_size, pop_dim, removal_size, removal_type, generation_type):
    # create initial reference population, and population
    ref_pop, pop = create_pop(ref_pop_size, pop_size, pop_dim, generation_type)

    # verify that the correct number of individuals were removed
    assert(len(ref_pop) == ref_pop_size)
    assert(len(pop) == pop_size)

    # # plot the reference pop
    # fig = plt.figure(figsize=(5, 5))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(ref_pop[:, 0], ref_pop[:, 1], ref_pop[:, 2], label='reference')
    # ax.scatter(pop[:, 0], pop[:, 1], pop[:, 2], label='new')
    # ax.set_facecolor("#ffebb8")
    # ax.set_title('Exemple distribution', fontsize=15)
    # plt.show()

    # compute initial ranking
    ranking_before_removal, novelties_before = compute_ranking(ref_pop, pop)

    # remove individuals inside reference population
    ref_pop, removal_time = remove_ind(ref_pop, removal_size, removal_type)

    # verify that the correct number of individuals were removed
    assert(len(ref_pop) == (ref_pop_size - removal_size))

    # compute ranking after removal of individuals inside reference population
    ranking_after_removal, novelties_after = compute_ranking(ref_pop, pop)

    # compare the two rankings
    ranking_distance = compare_rankings(ranking_before_removal, ranking_after_removal)[0]

    # compare the two novelty values
    nov_distance = LA.norm(novelties_after - novelties_before)

    return ranking_distance, removal_time, nov_distance


def add_method(ax, ref_pop_size, pop_sizes, pop_dim, color, method, n_exp, i, removal_ratio):

    removal_size = int(ref_pop_size * removal_ratio)
    ranking_distance_means = []
    ranking_distance_stds = []
    nov_distance_means = []
    nov_distance_stds = []
    removal_time_means = []
    for pop_size in tqdm.tqdm(pop_sizes):

        distances = []
        removal_times = []
        novelty_distances = []
        for _ in range(n_exp):
            ranking_distance, removal_time, novelty_distance = analyze(ref_pop_size, pop_size,
                                                                       pop_dim, removal_size, method,
                                                                       generation)
            distances.append(ranking_distance)
            novelty_distances.append(novelty_distance)
            removal_times.append(removal_time)
        distances = np.array(distances)
        mean = np.mean(distances)
        ranking_distance_means.append(mean)
        ranking_distance_stds.append([mean - np.percentile(distances, 25), np.percentile(distances, 75) - mean])
        
        distances = np.array(novelty_distances)
        mean = np.mean(distances)
        nov_distance_means.append(mean)
        nov_distance_stds.append([mean - np.percentile(distances, 25), np.percentile(distances, 75) - mean])
        removal_time_means.append(np.mean(np.array(removal_times)))
    pop_sizes = np.array(pop_sizes)

    ax[0].errorbar(pop_sizes + i * 0.6, ranking_distance_means,
                   np.transpose(ranking_distance_stds),
                   label=method,
                   fmt='-o', color=color, elinewidth=4)
    ax[1].errorbar(pop_sizes + i * 0.6, nov_distance_means,
                   np.transpose(nov_distance_stds),
                   label=method,
                   fmt='-o', color=color, elinewidth=4)
    ax[2].plot(pop_sizes, removal_time_means,
               label=method, color=color)


k = 1
removal_ratio = 0.3
ref_pop_size = 50 * k
pop_sizes = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]) * k
pop_dim = 3
n_exp = 100
generation = 'mixture'
methods = ['least_novel_iter', 'least_novel', 'random', 'gmm_sampling', 'grid', 'grid_density',
           'most_novel', 'most_novel_iter']

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(20, 10))
for i, method in enumerate(methods):
    add_method(ax, ref_pop_size, pop_sizes, pop_dim, utils.color_list[i], method, n_exp, i, removal_ratio)

ax[2].set_xlabel("Population size", labelpad=15, fontsize=12, color="#333533")
ax[0].set_facecolor("#ffebb8")
ax[0].set_title("Kendall Tau similarity between rankings", fontsize=15)
ax[0].legend(loc=4)
ax[1].set_facecolor("#ffebb8")
ax[1].set_title("Euclidean distance between novelty values", fontsize=15)
ax[1].legend(loc=4)
ax[2].set_facecolor("#ffebb8")
ax[2].set_ylabel("Time (s)", labelpad=15, fontsize=12, color="#333533")
ax[2].set_title("Removal computational time", fontsize=15)
ax[2].legend(loc=4)
plt.title('Archive size: ' + str(ref_pop_size))
plt.show()
