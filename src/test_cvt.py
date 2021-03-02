import utils
import numpy as np

bounds = [[1, 2], [1, 3]]
bounds = np.array(bounds)

samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (10000, len(bounds)))

a = utils.CVT(num_centroids=1000, bounds=bounds)

grid = np.zeros(1000)
for sample in samples:
    grid_idx = a.get_grid_index(sample)
    grid[grid_idx] += 1

coverage = np.count_nonzero(grid) / 1000
print(coverage)
