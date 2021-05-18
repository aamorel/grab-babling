import math
from scipy import interpolate
import numpy as np
import torch.nn as nn
import torch
import utils
from DynamicMovementPrimitives import PoseDMP
from pyquaternion import Quaternion

# ######################################### OPEN LOOP CONTROLLERS #####################################################


class DiscreteKeyPoints():

    def __init__(self, individual, info):
        """Share the time equally between keypoints

        """
        actions = []
        for i in range(info['NB_KEYPOINTS']):
            actions.append(individual[info['GENE_PER_KEYPOINTS'] * i:info['GENE_PER_KEYPOINTS'] * (i + 1)])
        self.actions = actions
        self.n_iter_per_action = math.ceil(info['n_iter'] / len(actions))
        self.open_loop = True
        self.initial_action = actions[0]

    def get_action(self, i):
        action_index = i // self.n_iter_per_action
        action = self.actions[action_index]
        return action
        

class InterpolateKeyPoints():

    def __init__(self, individual, info, initial=None):
        """Interpolate actions between keypoints

        """
        actions = []
        for i in range(info['NB_KEYPOINTS']):
            actions.append(individual[info['GENE_PER_KEYPOINTS'] * i:info['GENE_PER_KEYPOINTS'] * (i + 1)])
        self.actions = actions
        n_keypoints = len(actions)
        interval_size = int(info['n_iter'] / n_keypoints)
        interp_x = [int(interval_size / 2 + i * interval_size) for i in range(n_keypoints)]
        if initial is not None: # add initial joint states to get a smooth motion
            actions.insert(0, initial[0][:-1]) # the gripper is not included
            interp_x.insert(0, 0)
        self.action_polynome = interpolate.interp1d(interp_x, actions, kind='quadratic', axis=0,
                                                    bounds_error=False, fill_value='extrapolate')
        self.open_loop = True
        self.initial_actions = actions[0]
        self.grip_time = int(additional_gene / 2 + 0.5)

    def get_action(self, i):
        action = self.action_polynome(i)
        action = np.append(action, (i<self.grip_time)*2-1)
        return action


class InterpolateKeyPointsEndPause():

    def __init__(self, individual, info, initial=None):
        """Interpolate actions between keypoints
           Stops the movement at the end in order to make sure that the object was correctly grasped at the end.

        """

        actions = []
        for i in range(info['NB_KEYPOINTS']):
            actions.append(individual[info['GENE_PER_KEYPOINTS'] * i:info['GENE_PER_KEYPOINTS'] * (i + 1)])

        n_keypoints = len(actions)
        self.pause_time = info['pause_frac'] * info['n_iter']
        interval_size = int(self.pause_time / n_keypoints)
        interp_x = [int(interval_size / 2 + i * interval_size) for i in range(n_keypoints)]
        if initial is not None: # add initial joint states to get a smooth motion
            actions.insert(0, initial[0][:-1]) # the gripper is not included
            interp_x.insert(0, 0)
        self.action_polynome = interpolate.interp1d(interp_x, actions, kind='quadratic', axis=0,
                                                    bounds_error=False, fill_value='extrapolate')
        self.open_loop = True
        self.last_action = 0
        self.initial_action = actions[0]

    def get_action(self, i):
        if i <= self.pause_time:
            action = self.action_polynome(i)
            self.last_action = action
        else:
            action = self.last_action
        return action


class InterpolateKeyPointsEndPauseGripAssumption():

    def __init__(self, individual, info, initial=None):
        """Interpolate actions between keypoints
           Stops the movement at the end in order to make sure that the object was correctly grasped at the end.
           Only one parameter for the gripper, specifying the time at which it should close

        """
        actions = []
        gene_per_key = info['GENE_PER_KEYPOINTS'] - 1
        for i in range(info['NB_KEYPOINTS']):
            actions.append(individual[gene_per_key * i:gene_per_key * (i + 1)])

        additional_gene = individual[-1]

        n_keypoints = len(actions)
        self.pause_time = info['pause_frac'] * info['n_iter']
        interval_size = int(self.pause_time / n_keypoints)
        interp_x = [int(interval_size / 2 + i * interval_size) for i in range(n_keypoints)]
        if initial is not None: # add initial joint states to get a smooth motion
            actions.insert(0, initial[0][:-1]) # the gripper is not included
            interp_x.insert(0, 0)
        self.action_polynome = interpolate.interp1d(interp_x, actions, kind='quadratic', axis=0,
                                                    bounds_error=False, fill_value='extrapolate')
        self.open_loop = True
        self.last_action = 0
        self.grip_time = int((additional_gene / 2 + 0.5) * self.pause_time)
        self.initial_action = actions[0]
        self.initial_action = np.append(self.initial_action, 1)

    def get_action(self, i):
        if i <= self.pause_time:
            action = self.action_polynome(i)
            if i < self.grip_time:
                action = np.append(action, 1)  # gripper is open
            else:
                action = np.append(action, -1)  # gripper is closed
            self.last_action = action
        else:
            # feed last action
            action = self.last_action
        return action


class DMPGripLift():
    def __init__(self, ind, info, initial):
        self.n_iter, self.rollout = info['n_iter'], info['n_rollout']
        self.goalQuaternion, self.liftQuaternion = Quaternion(ind[3:7]).normalised, Quaternion(ind[7:11]).normalised
        τ = info['τ'] if 'τ' in info.keys() else 1
        self.dmp = PoseDMP(start=initial[1][0], goal=ind[:3], weights=np.array(ind[11:]).reshape(3,-1), startQuaternion=initial[1][1], goalQuaternion=ind[3:7], weightsQuaternion=2, τ=τ, αz=4, αyx=1000, phaseStopping=True)
        self.grip_time = float('inf')
        self.initial_action = initial[0]
        self.open_loop = False
        self.liftTime = 2

    def get_action(self, i, observation):
        end_effetor_pose = observation[2]+[observation[3][j] for j in (3,0,1,2)]
        for i in range(self.rollout): pose = self.dmp.step(end_effetor_pose)
        #print("goal error",self.dmp.distance(observation[2], self.dmp.goal), "current error", self.dmp.distance(observation[2], pose[0]),  "goal", self.dmp.goal, "command", pose[0], "current", observation[2])
        if self.dmp.distance(observation[2], self.dmp.goal) < 1e-3:
            self.get_action = self.lift
            self.vertical_inter = interpolate.interp1d(np.array([i, i+self.liftTime*240/self.rollout]), np.array([self.dmp.goal[2], self.dmp.goal[2]+0.1]), kind='linear', bounds_error=False, fill_value='extrapolate')
            self.quaternion_inter = Quaternion.intermediates(Quaternion(self.goalQuaternion), Quaternion(self.liftQuaternion), n=int(self.liftTime*240/self.rollout), include_endpoints=True)
            self.grip_time = i
        return {'position': pose[0], 'quaternion':pose[1], 'gripper close':False}
        
    def lift(self, i, observation):

        return {'position': [self.dmp.goal[0], self.dmp.goal[1], self.vertical_inter(i)], 'quaternion':next(self.quaternion_inter, self.liftQuaternion), 'gripper close':True}
        


# ######################################### CLOSE LOOP CONTROLLERS #####################################################


class ClosedLoopEndPauseGripAssumption():
    def __init__(self, individual, info):
        self.pause_time = info['pause_frac'] * info['n_iter']
        additional_gene = individual[-1]
        self.grip_time = math.floor((additional_gene / 2 + 0.5) * self.pause_time)
        self.open_loop = False
        # for now, random
        self.initial_action = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.last_action = self.initial_action
        self.gain = 0.001
        self.action_bounds = [[-0.999, 0.999]] * 7

        # network
        # assuming 34 inputs
        assert(len(individual) == 344)  # counting the additional gene
        self.l1 = nn.Linear(34, 8)
        weight_1 = nn.Parameter(torch.Tensor(individual[:272]).reshape((8, 34)))
        bias_1 = nn.Parameter(torch.Tensor(individual[272:280]).reshape(8))
        self.l1.weight = weight_1
        self.l1.bias = bias_1
        self.l2 = nn.Linear(8, 7)
        weight_2 = nn.Parameter(torch.Tensor(individual[280:336]).reshape((7, 8)))
        bias_2 = nn.Parameter(torch.Tensor(individual[336:343]).reshape(7))
        self.l2.weight = weight_2
        self.l2.bias = bias_2
        self.r1 = nn.ReLU()
        self.r2 = nn.Tanh()
        self.action_network = nn.Sequential(self.l1, self.r1, self.l2, self.r2)

    def get_action(self, i, obs):
        if i <= self.pause_time:
            input_to_net = np.array(obs[0] + obs[1] + obs[2] + obs[3] + list(obs[4]) + [i / self.pause_time])
            with torch.no_grad():
                input_to_net = torch.Tensor(input_to_net)
                action = self.action_network(input_to_net).numpy()

                # hypothesis: control in speed instead of position
                action = self.last_action[:7] + self.gain * action
                
                utils.bound(action, self.action_bounds)
                
            if i < self.grip_time:
                action = np.append(action, 1)  # gripper is open
            else:
                action = np.append(action, -1)  # gripper is closed
            self.last_action = action
        else:
            # feed last action
            action = self.last_action
        return action
