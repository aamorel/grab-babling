import pyquaternion as pyq
import numpy as np
import math
from sklearn.cluster import SpectralClustering as SC
from sklearn.neighbors import NearestNeighbors as Nearest
import utils
import pybullet as p
import pybullet_data

n_samples = 1000
K = 20
n_clusts = [10, 20, 30, 40, 50, 60, 70, 80, 90]
n_clust = 20
n_samples_arr = [100, 500, 1000, 2000, 4000]
n_test = 500

sqrt_2 = math.sqrt(2)

dists = []
for n_clust in n_clusts:
    quats = []
    quats_arr = []
    for _ in range(n_samples):
        a = pyq.Quaternion.random()
        a = a.normalised
        quats.append(a)
        quats_arr.append(a.elements)

    quats_arr = np.array(quats_arr)

    # compute similarity matrix
    X = np.zeros((n_samples, n_samples))
    for x in range(n_samples):
        for y in range(n_samples):
            if x == y:
                X[x, y] = 1
            else:
                a = quats[x]
                b = quats[y]
                X[x, y] = (sqrt_2 - pyq.Quaternion.absolute_distance(a, b)) / sqrt_2

    clustering = SC(n_clusters=n_clust, affinity='precomputed')
    clustering.fit(X)
    samples_labels = clustering.labels_
    print(np.bincount(samples_labels))

    neigh = Nearest(n_neighbors=K, metric=utils.quatmetric)
    neigh.fit(quats_arr)

    pos = []
    labels = []
    for _ in range(n_test):
        quat = pyq.Quaternion.random()
        test = np.array(quat.elements)
        neigh_indices = neigh.kneighbors(test.reshape(1, -1))[1][0]
        neigh_labels = samples_labels[neigh_indices]
        label = np.bincount(neigh_labels).argmax()
        pos.append(quat)
        labels.append(label)

    mean_distances = []
    for i in range(n_clust):
        count = 0
        j = 0
        members = []
        while count <= 5 and j < n_test:
            if labels[j] == i:
                members.append(pos[j])
                count += 1

            j += 1
        distances = []
        for a in members:
            for b in members:
                distances.append(pyq.Quaternion.absolute_distance(a, b))
        distances = np.array(distances)
        if len(distances) > 0:
            mean = np.mean(distances)
            mean_distances.append(mean)
    mean_distances = np.array(mean_distances)
    dists.append(np.mean(mean_distances))
    print('finished experiment')

print(dists)

# physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
# p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
# p.setGravity(0, 0, 0)
# planeId = p.loadURDF("plane.urdf")
# cubeStartPos = [0, 0, 2]
# cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
# boxId = p.loadURDF("r2d2.urdf", cubeStartPos, cubeStartOrientation)

# for i in range(n_clust):
#     count = 0
#     j = 0
#     while count <= 5:
#         if labels[j] == i:
#             p.resetBasePositionAndOrientation(boxId, cubeStartPos, [pos[j][1], pos[j][2], pos[j][3], pos[j][0]])
#             print(i)
#             time.sleep(2)
#             count += 1
#
#         j += 1
# p.disconnect()
