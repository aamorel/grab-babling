import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import distance
from numpy import linalg as LA
import pybullet_envs
import pybullet_envs.gym_locomotion_envs
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset
import ruamel.yaml

yaml = ruamel.yaml.YAML()
yaml.width = 10000 # this is the output line width after which wrapping occurs

color_list = ["green", "blue", "red",
              "#FFFF00", "#1CE6FF", "#FF34FF", "#006FA6", "#A30059",
              "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
              "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
              "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
              "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
              "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
              "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
              "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
              "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
              "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
              "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
              "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
              "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
              "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
              "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
              "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"]


class CVT():
    def __init__(self, num_centroids=7, bounds=[[-1, 1]], num_samples=100000,
                 num_replicates=1, max_iterations=20, tolerance=0.001, sphere=False):
        
        self.num_centroids = num_centroids
        self.bounds = bounds
        self.low, self.high = np.array(bounds).T
        self.num_samples = num_samples
        self.num_replicates = num_replicates
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        X = np.random.default_rng().random(size=(num_samples, len(bounds)))
        if sphere: # normalize all samples to get a sphere
            X /= np.linalg.norm(X, axis=-1)[:,None]
        
        kmeans = KMeans(init='k-means++',
                        n_clusters=num_centroids,
                        n_init=num_replicates,
                        max_iter=max_iterations,
                        tol=tolerance,
                        verbose=0)
        
        kmeans.fit(X)

        self.centroids = kmeans.cluster_centers_
        self.k_tree = KDTree(self.centroids)

    def get_grid_index(self, sample):
        # map back to [-1, 1]
        sample_copy = (np.array(sample)-self.low) / (self.high-self.low) * 2 - 1
        grid_index = self.k_tree.query(sample_copy, k=1)[1]
        return grid_index


def bound(behavior, bound_behavior):
    for i in range(len(behavior)):
        if behavior[i] < bound_behavior[i][0]:
            behavior[i] = bound_behavior[i][0]
        if behavior[i] > bound_behavior[i][1]:
            behavior[i] = bound_behavior[i][1]


def normalize(behavior, bound_behavior):
    for i in range(len(behavior)):
        range_of_interval = bound_behavior[i][1] - bound_behavior[i][0]
        mean_of_interval = (bound_behavior[i][0] + bound_behavior[i][1]) / 2
        behavior[i] = (behavior[i] - mean_of_interval) / (range_of_interval / 2)


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
    uniformity = 1 - distance.jensenshannon(P, Q)
    return uniformity


def quatmetric(a, b):
    """Takes as input two 4-dimensional arrays representing quaternion and returns a significative
       distance between the two orientations.
       Inspired by 'Comparing Distance Metrics for Rotation Using the k-Nearest Neighbors Algorithm for
       Entropy Estimation' and PyQuaternion.

    Args:
        a (ndarray): first quaternion
        b (ndarray): second quaternion

    Returns:
        float: distance between 0 and sqrt(2)
    """
    aminusb = a - b
    aplusb = a + b
    normminus = LA.norm(aminusb)
    normplus = LA.norm(aplusb)
    
    if normminus < normplus:
        return normminus
    else:
        return normplus


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)
    

def circle_coordinates(n_rep, r):
    coordinates = []
    delta_rad = 2 * math.pi / n_rep
    for i in range(n_rep):
        theta = i * delta_rad
        x = math.cos(theta) * r
        y = math.sin(theta) * r
        c = [x, y]
        coordinates.append(c)

    return coordinates


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def half_sphere_projection(r=1, num=10):
    linspace = np.linspace(-r, r, num=num)
    points = np.array(np.meshgrid(linspace, linspace)).T.reshape(-1, 2)
    points = points[np.linalg.norm(points, axis=1) <= r]
    return np.hstack([points, np.sqrt(r * r - (points[:, 0] * points[:, 0] + points[:, 1] * points[:, 1]))[:, None]])


class NeuralAgentNumpy():
    def __init__(self, n_in, n_out, n_hidden_layers=2, n_neurons_per_hidden=5):
        self.dim_in = n_in
        self.dim_out = n_out
        self.n_per_hidden = n_neurons_per_hidden
        self.n_hidden_layers = n_hidden_layers
        self.weights = None
        self.n_weights = None
        self.bias = None
        self.opt_state = None
        self.out = np.zeros(n_out)
    
    def randomize(self):
        if self.n_hidden_layers > 0:
            self.weights = [2 * np.random.random((self.dim_in, self.n_per_hidden)) - 1]
            self.bias = [2 * np.random.random(self.n_per_hidden) - 1]  # In -> first hidden
            for i in range(self.n_hidden_layers - 1):  # Hidden -> hidden
                self.weights.append(2 * np.random.random((self.n_per_hidden, self.n_per_hidden)) - 1)
                self.bias.append(2 * np.random.random(self.n_per_hidden) - 1)
            self.weights.append(2 * np.random.random((self.n_per_hidden, self.dim_out)) - 1)  # -> last hidden -> out
            self.bias.append(2 * np.random.random(self.dim_out) - 1)
        else:
            self.weights = [2 * np.random.random((self.dim_in, self.dim_out)) - 1]  # Single-layer perceptron
            self.bias = [2 * np.random.random(self.dim_out) - 1]
        n_weights_1 = np.sum([np.product(w.shape) for w in self.weights])
        self.n_weights = n_weights_1 + np.sum([np.product(b.shape) for b in self.bias])
        
    def get_weights(self):
        """
        Returns all network parameters as a single array
        """
        flat_weights = np.hstack([arr.flatten() for arr in (self.weights + self.bias)])
        return flat_weights

    def set_weights(self, flat_parameters):
        """
        Set all network parameters from a single array
        """
        if np.nan in flat_parameters:
            print("WARNING: NaN in the parameters of the NN: " + str(list(flat_parameters)))
        if max(flat_parameters) > 1000:
            print("WARNING: max value of the parameters of the NN >1000: " + str(list(flat_parameters)))

        i = 0  # index
        self.weights = list()
        self.bias = list()
        if self.n_hidden_layers > 0:
            # In -> first hidden
            w0 = np.array(flat_parameters[i:(i + self.dim_in * self.n_per_hidden)])
            self.weights.append(w0.reshape(self.dim_in, self.n_per_hidden))
            i += self.dim_in * self.n_per_hidden
            for _ in range(self.n_hidden_layers - 1):  # Hidden -> hidden
                w = np.array(flat_parameters[i:(i + self.n_per_hidden * self.n_per_hidden)])
                self.weights.append(w.reshape((self.n_per_hidden, self.n_per_hidden)))
                i += self.n_per_hidden * self.n_per_hidden
            # -> last hidden -> out
            wN = np.array(flat_parameters[i:(i + self.n_per_hidden * self.dim_out)])
            self.weights.append(wN.reshape((self.n_per_hidden, self.dim_out)))
            i += self.n_per_hidden * self.dim_out
            # Samefor bias now
            # In -> first hidden
            b0 = np.array(flat_parameters[i:(i + self.n_per_hidden)])
            self.bias.append(b0)
            i += self.n_per_hidden
            for _ in range(self.n_hidden_layers - 1):  # Hidden -> hidden
                b = np.array(flat_parameters[i:(i + self.n_per_hidden)])
                self.bias.append(b)
                i += self.n_per_hidden
            # -> last hidden -> out
            bN = np.array(flat_parameters[i:(i + self.dim_out)])
            self.bias.append(bN)
            i += self.dim_out
        else:
            n_w = self.dim_in * self.dim_out
            w = np.array(flat_parameters[:n_w])
            self.weights = [w.reshape((self.dim_in, self.dim_out))]
            self.bias = [np.array(flat_parameters[n_w:])]
        n_weights_1 = np.sum([np.product(w.shape) for w in self.weights])
        self.n_weights = n_weights_1 + np.sum([np.product(b.shape) for b in self.bias])
    
    def choose_action(self, x):
        """
        Propagate
        """
        if(self.n_hidden_layers > 0):
            # Input
            y = np.matmul(x, self.weights[0]) + self.bias[0]
            y = tanh(y)
            # hidden -> hidden
            for i in range(1, self.n_hidden_layers - 1):
                y = np.matmul(y, self.weights[i]) + self.bias[i]
                y = tanh(y)
            # Out
            a = np.matmul(y, self.weights[-1]) + self.bias[-1]
            out = tanh(a)
            return out
        else:  # Simple monolayer perceptron
            return tanh(np.matmul(x, self.weights[0]) + self.bias[0])

    def get_opt_state(self):
        return self.opt_state

    def set_opt_state(self, state):
        self.opt_state = state

    def __getstate__(self):
        dic = dict()
        dic["dim_in"] = self.dim_in
        dic["dim_out"] = self.dim_out
        dic["n_hidden_layers"] = self.n_hidden_layers
        dic["n_per_hidden"] = self.n_per_hidden
        dic["opt_state"] = self.opt_state
        dic["as_vector"] = self.get_weights()
        return dic

    def __setstate__(self, dic):
        self.__init__(dic["dim_in"], dic["dim_out"], dic["n_hidden_layers"], dic["n_per_hidden"])
        self.set_weights(dic["as_vector"])
        self.set_opt_state(dic["opt_state"])


class DeterministicPybulletAnt(pybullet_envs.gym_locomotion_envs.AntBulletEnv):
    def __init__(self, render=False, random_seed=0):
        self.deterministic_random_seed = random_seed
        pybullet_envs.gym_locomotion_envs.AntBulletEnv.__init__(self, render)

    def reset(self):
        self.seed(self.deterministic_random_seed)
        return pybullet_envs.gym_locomotion_envs.AntBulletEnv.reset(self)


class BDDataset(Dataset):

    def __init__(self, bds):

        self.bds = bds

    def __len__(self):
        return len(self.bds)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.bds[idx]

        return np.array(sample)


class AE(nn.Module):
    def __init__(self, input_shape, n_hidden, n_reduced_dim):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=input_shape, out_features=n_hidden
        )
        self.encoder_output_layer = nn.Linear(
            in_features=n_hidden, out_features=n_reduced_dim
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=n_reduced_dim, out_features=n_hidden
        )
        self.decoder_output_layer = nn.Linear(
            in_features=n_hidden, out_features=input_shape
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.sigmoid(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.tanh(activation)
        return reconstructed
    
    def encode(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.sigmoid(code)
        return code

def setFlowStyle(seq): # represent seq in flow style in the yaml file
    if isinstance(seq, np.ndarray) and np.ndim(seq)==0:
        return seq.item()
    s = ruamel.yaml.comments.CommentedSeq(seq)
    s.fa.set_flow_style()
    return s
    
def save_yaml(data:dict, path:str):
    for key, value in data.items(): # setup yaml
        if isinstance(value, (tuple, list, np.ndarray)):
            data[key] = setFlowStyle(value)
    with open(path, 'w') as f:
        yaml.dump(data, f)
