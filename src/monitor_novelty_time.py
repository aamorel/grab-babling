from sklearn.neighbors import NearestNeighbors as Nearest
import numpy as np
import time as tm
import matplotlib.pyplot as plt
K = 15


def compute_average_distance(query, k_tree):

    neighbours_distances = k_tree.kneighbors(X=query)[0][:, 1:]

    avg_distance = np.mean(neighbours_distances, axis=1)
    return avg_distance


def assess_novelties(pop, archive, algo_type):

    reference_pop = np.concatenate((pop, archive), axis=0)

    k_tree = Nearest(n_neighbors=K + 1, algorithm=algo_type)
    k_tree.fit(reference_pop)
    # compute novelty for current individuals (loop only on the pop)
    a = tm.time()
    novelties = compute_average_distance(pop, k_tree)
    b = tm.time()
    c = b - a
    # k_tree = Nearest(n_neighbors=K + 1, algorithm='brute')
    # k_tree.fit(reference_pop)
    # # compute novelty for current individuals (loop only on the pop)
    # novelties_1 = compute_average_distance(pop, k_tree)
    
    # k_tree = Nearest(n_neighbors=K + 1, algorithm='ball_tree')
    # k_tree.fit(reference_pop)
    # # compute novelty for current individuals (loop only on the pop)
    # novelties_2 = compute_average_distance(pop, k_tree)
    
    # k_tree = Nearest(n_neighbors=K + 1, algorithm='kd_tree')
    # k_tree.fit(reference_pop)
    # # compute novelty for current individuals (loop only on the pop)
    # novelties_3 = compute_average_distance(pop, k_tree)

    return novelties, c


def analyse_time(pop_size, archive_size, behavior_size, algo_type, archive_generation_type):
    pop = np.random.rand(pop_size, behavior_size)
    if archive_generation_type == 'uniform':
        archive = np.random.rand(archive_size, behavior_size)
    if archive_generation_type == 'multi':
        means = np.random.rand(behavior_size)
        covs = np.random.rand(behavior_size) / 30
        covs = np.diag(covs)
        archive = np.random.multivariate_normal(means, covs, size=(archive_size,))
        archive = np.clip(archive, 0, 1)
    if archive_generation_type == 'structured':
        archive = np.zeros((archive_size, behavior_size))
        archive[:, 0] = np.random.rand(archive_size)
    time_in = tm.time()
    _, query_time = assess_novelties(pop, archive, algo_type)
    time_out = tm.time()
    time = time_out - time_in
    return query_time


# bar_width = 0.5
# pop_size = 10
# archive_generation_type = 'uniform'
# archive_sizes = [1000, 10000, 100000, 1000000]
# behavior_sizes = [5, 10, 20, 40]
# algo_types = ['ball_tree', 'kd_tree', 'brute']
# n_exp = 5
# for algo_type in algo_types:
#     fig, ax = plt.subplots(1, 1, figsize=(15, 10))
#     b = []
#     for i, behavior_size in enumerate(behavior_sizes):
#         means = []
#         bars_n = len(archive_sizes)
#         for archive_size in archive_sizes:
#             t_s = []
#             for _ in range(n_exp):
#                 t = analyse_time(pop_size, archive_size, behavior_size, algo_type, archive_generation_type)
#                 t_s.append(t)
#             t_s = np.array(t_s)
#             mean = np.mean(t_s)
#             means.append(mean)
#         offset = bar_width * bars_n + 1
#         place_offset = i * offset
#         bars = np.ones(bars_n) * place_offset
#         for j, bar in enumerate(bars):
#             bars[j] += j * bar_width
#         b.append(bars)
#         ax.bar(bars, means, width=bar_width, label='Behavior descriptor size = ' + str(behavior_size))
#     ax.legend()
#     ax.set_yscale('log')
#     ax.set_ylabel('Novelty computational time')
#     ax.set_xlabel('Archive size')
#     ax.set_title('Novelty computational time with ' + str(algo_type) + ' algorithm')
#     labels = archive_sizes * len(behavior_sizes)
#     b = np.array(b)
#     b = b.flatten()
#     plt.xticks(b, labels, rotation=90)
#     plt.show()
#     fig.savefig(str(algo_type) + 'study.png')


bar_width = 0.5
pop_size = 100
archive_generation_type = 'uniform'
archive_sizes = [1000, 10000, 100000]
behavior_size = 5
algo_types = ['ball_tree', 'kd_tree', 'brute']
n_exp = 5
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
b = []
for i, algo_type in enumerate(algo_types):
    means = []
    bars_n = len(archive_sizes)
    for archive_size in archive_sizes:
        t_s = []
        for _ in range(n_exp):
            t = analyse_time(pop_size, archive_size, behavior_size, algo_type, archive_generation_type)
            t_s.append(t)
        t_s = np.array(t_s)
        mean = np.mean(t_s)
        means.append(mean)
    offset = bar_width * bars_n + 1
    place_offset = i * offset
    bars = np.ones(bars_n) * place_offset
    for j, bar in enumerate(bars):
        bars[j] += j * bar_width
    b.append(bars)
    ax.bar(bars, means, width=bar_width, label='Algo type: ' + str(algo_type))
ax.legend()
ax.set_yscale('log')
ax.set_ylabel('Novelty computational time (s)')
ax.set_xlabel('Archive size')
ax.set_title('Novelty computational time with respect to archive size')
labels = archive_sizes * len(algo_types)
b = np.array(b)
b = b.flatten()
plt.xticks(b, labels, rotation=90)
plt.show()
# fig.savefig(str(algo_type) + 'study.png')
