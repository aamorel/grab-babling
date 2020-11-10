import numpy as np
import random
import matplotlib.pyplot as plt
import math
N = 20000


def distance(p1, p2):

    dist = 0
    for i, dim in enumerate(p1):
        dist += (dim - p2[i])**2
    return dist ** (1 / 2)


def eval(set_bounds):
    # initialize possible set size
    possible_set_size = 1
    for dim in set_bounds:
        possible_set_size *= dim[1] - dim[0]
    possible_set_size = math.log(possible_set_size)

    # sample N distances
    distances = []
    for i in range(N):
        p1 = []
        p2 = []
        for dim in set_bounds:
            p1.append(random.uniform(dim[0], dim[1]))
            p2.append(random.uniform(dim[0], dim[1]))
        distances.append(distance(p1, p2))

    mean_distance = np.mean(np.array(distances))
    return possible_set_size, mean_distance


fig, ax = plt.subplots()
for n_dim in [2, 3, 4, 5, 6, 7, 8]:
    possibles = []
    means_dist = []
    for n_exp in range(100):
        set_bounds = []
        for i in range(n_dim):
            a = random.uniform(-10, 10)
            b = random.uniform(-10, 10)
            if a >= b:
                set_bounds.append([b, a])
            else:
                set_bounds.append([a, b])
        pos, mean = eval(set_bounds)
        possibles.append(pos)
        means_dist.append(mean)

    possibles = np.array(possibles)
    means_dist = np.array(means_dist)

    ax.scatter(possibles, means_dist, label=n_dim)

plt.legend()
plt.show()
