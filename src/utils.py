import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import distance


class CVT():
    def __init__(self, num_centroids=7, bounds=[[-1, 1]], num_samples=100000,
                 num_replicates=1, max_iterations=100, tolerance=0.001):
        
        self.num_centroids = num_centroids
        self.bounds = bounds
        self.num_samples = num_samples
        self.num_replicates = num_replicates
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        X = []
        for bound in bounds:
            X.append(np.random.uniform(low=bound[0], high=bound[1], size=self.num_samples))
        X = np.array(X)
        X = np.transpose(X)

        kmeans = KMeans(init='k-means++',
                        n_clusters=num_centroids,
                        n_init=num_replicates,
                        n_jobs=-1,
                        max_iter=max_iterations,
                        tol=tolerance,
                        verbose=0)
        
        kmeans.fit(X)

        self.centroids = kmeans.cluster_centers_
        self.k_tree = KDTree(self.centroids)

    def get_grid_index(self, sample):
        grid_index = self.k_tree.query(sample, k=1)[1]
        return grid_index


def bound(behavior, bound_behavior):
    for i in range(len(behavior)):
        if behavior[i] < bound_behavior[i][0]:
            behavior[i] = bound_behavior[i][0]
        if behavior[i] > bound_behavior[i][1]:
            behavior[i] = bound_behavior[i][1]


def list_l2_norm(list1, list2):
    if len(list1) != len(list2):
        raise NameError('The two lists have different length')
    else:
        dist = 0
        for i in range(len(list1)):
            dist += (list1[i] - list2[i]) ** 2
        dist = dist ** (1 / 2)
        return dist


def compute_uniformity(grid):
    P = grid[np.nonzero(grid)]
    P = P / np.sum(P)
    Q = np.ones(len(P)) / len(P)
    uniformity = distance.jensenshannon(P, Q)
    return uniformity
