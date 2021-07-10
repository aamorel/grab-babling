import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import distance
from scipy import interpolate
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.distributions import DiagGaussianDistribution, SquashedDiagGaussianDistribution, CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution, StateDependentNoiseDistribution
import torch as th
import inspect
import functools
import os
import contextlib
import sys
import io

color_list = ["#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
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
                 num_replicates=1, max_iterations=20, tolerance=0.001):
        
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

    
class PDControllerStable(object):
  """
  Implementation based on: Tan, J., Liu, K., & Turk, G. (2011). "Stable proportional-derivative controllers"
  DOI: 10.1109/MCG.2011.30
  """
  def __init__(self, pb):
    self._pb = pb

  def computePD(self, bodyUniqueId, jointIndices, desiredPositions, desiredVelocities, kps, kds, maxForces, timeStep):
    jointStates = self._pb.getJointStates(bodyUniqueId, jointIndices)
    q1 = []
    qdot1 = []
    zeroAccelerations = []
    for i in range(len(jointIndices)):
      q1.append(jointStates[i][0])
      qdot1.append(jointStates[i][1])
      zeroAccelerations.append(0)

    q = np.array(q1)
    qdot = np.array(qdot1)
    qdes = np.array(desiredPositions)
    qdotdes = np.array(desiredVelocities)

    qError = qdes - q
    qdotError = qdotdes - qdot

    Kp = np.diagflat(kps)
    Kd = np.diagflat(kds)

    # Compute -Kp(q + qdot - qdes)
    p_term = Kp.dot(qError - qdot*timeStep)
    # Compute -Kd(qdot - qdotdes)
    d_term = Kd.dot(qdotError)

    # Compute Inertia matrix M(q)
    M = self._pb.calculateMassMatrix(bodyUniqueId, q1)
    M = np.array(M)
    # Given: M(q) * qddot + C(q, qdot) = T_ext + T_int
    # Compute Coriolis and External (Gravitational) terms G = C - T_ext
    G = self._pb.calculateInverseDynamics(bodyUniqueId, q1, qdot1, zeroAccelerations)
    G = np.array(G)
    # Obtain estimated generalized accelerations, considering Coriolis and Gravitational forces, and stable PD actions
    qddot = np.linalg.solve(a=(M + Kd * timeStep),
                            b=(-G + p_term + d_term))
    # Compute control generalized forces (T_int)
    tau = p_term + d_term - (Kd.dot(qddot) * timeStep)
    # Clip generalized forces to actuator limits
    maxF = np.array(maxForces)
    generalized_forces = np.clip(tau, -maxF, maxF)
    return generalized_forces

class MLP(BaseModel):
    def __init__(self, *args, net_arch=[32,32], output_transform=None, **kwargs):
        super(MLP, self).__init__(*args, **kwargs)

        self.features_extractor = self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.net_arch = net_arch
        self.output_transform = output_transform
        distributions = {
            "DiagGaussianDistribution": DiagGaussianDistribution,
            "SquashedDiagGaussianDistribution": SquashedDiagGaussianDistribution,
            "CategoricalDistribution": CategoricalDistribution,
            "MultiCategoricalDistribution": MultiCategoricalDistribution,
            "BernoulliDistribution": BernoulliDistribution,
            "StateDependentNoiseDistribution": StateDependentNoiseDistribution,
        }
        self.action_dim = get_action_dim(self.action_space)
        output_dim = self.action_dim
        if output_transform in distributions:
            self.distribution = distributions[output_transform](self.action_dim)
            if output_transform in {"DiagGaussianDistribution", "SquashedDiagGaussianDistribution"}: output_dim = self.action_dim*2
        else:
            self.distribution = None

        layers = create_mlp(input_dim=self.features_extractor.features_dim, output_dim=output_dim, net_arch=net_arch, activation_fn=th.nn.LeakyReLU)
        if output_transform is not None and output_transform not in distributions:
            layers += [getattr(th.nn, output_transform)() if isinstance(output_transform, str) else output_transform()]
        self.mlp = th.nn.Sequential(*layers)
        self.optimizer_kwargs['lr'] = self.optimizer_kwargs.get('lr', 5e-4)
        self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)
    
    
    def forward(self, data, deterministic=False):
        out = self.mlp(self.features_extractor(data))
        if self.distribution is None:
            return out
        elif self.output_transform in {"DiagGaussianDistribution", "SquashedDiagGaussianDistribution"}:
            mean, log_std = th.split(out, self.action_dim, dim=-1)
            return self.distribution.actions_from_params(mean_actions=mean, log_std=log_std, deterministic=deterministic)
        else:
            return self.distribution.actions_from_params(action_logits=out, deterministic=deterministic)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                output_transform=self.output_transform,
            )
        )
        return data

class InterpolateKeyPointsGrip():

    def __init__(self, individual, n_iter, genes_per_keypoint, nb_keypoints, initial=None):
        """Interpolate actions between keypoints
           Only one parameter (last gene) for the gripper, specifying the time at which it should close
        """
        assert len(individual) == nb_keypoints * genes_per_keypoint + 1, f"len(individual)={len(individual)} must be equal to nb_keypoints({nb_keypoints}) * genes_per_keypoint({genes_per_keypoint}) + 1(gripper) = {nb_keypoints * genes_per_keypoint + 1}"
        self.n_iter = n_iter
        actions = np.split(np.array(individual[:-1]), nb_keypoints)

        interval_size = int(n_iter / nb_keypoints)
        interp_x = [int(interval_size / 2 + i * interval_size) for i in range(nb_keypoints)]
        if initial is not None: # add initial joint states to get a smooth motion
            assert len(initial) == genes_per_keypoint, f"The length of initial={len(initial)} must be genes_per_keypoint={genes_per_keypoint}"
            actions.insert(0, initial) # the gripper is not included
            interp_x.insert(0, 0)
        self.action_polynome = interpolate.interp1d(interp_x, actions, kind='quadratic', axis=0,
                                                    bounds_error=False, fill_value='extrapolate')
        self.open_loop = True
        self.grip_time = int((individual[-1]+1)/2 * n_iter)

    def get_action(self, i, _o=None):
        if i <= self.n_iter:
            action = self.action_polynome(i)
            action = np.append(action, 1 if i < self.grip_time else -1)  # gripper: 1=open, -1=closed
            return action
        else: # feed last action
            return self.get_action(self.n_iter)
