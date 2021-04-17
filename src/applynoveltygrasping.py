import gym
import noveltysearch
import utils
import numpy as np
import pyquaternion as pyq
import matplotlib.pyplot as plt
from deap import base, creator
from sklearn.cluster import AgglomerativeClustering
import controllers
import os
import json
import glob
import random
import math
import time
import sys
import argparse
from functools import partial
from pathlib import Path

DISPLAY = False
PARALLELIZE = True
PLOT = True
DISPLAY_HOF = False
DISPLAY_RAND = False
DISPLAY_TRIUMPHANTS = False
EVAL_SUCCESSFULL = False
EVAL_WITH_OBSTACLE = False
EVAL_QUALITY = False
SAVE_TRAJ = False
SAVE_ALL = False
RESET_MODE = True

def greater(name, min, value):
    v = int(value)
    if v <= min: raise argparse.ArgumentTypeError(f"The {name.strip()} must be greater than {min}")
    return v

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--robot", help="The robot environment", type=str, default="baxter", choices=["baxter", "kuka", "pepper"])
parser.add_argument("-o", "--object", help="The object to grasp", type=str, default="sphere")
parser.add_argument("-p", "--population", help="The poulation size", type=partial(greater, "population size", 1), default=96)
parser.add_argument("-g", "--generation", help="The number of generation", type=partial(greater, "number of generation", 1), default=1000)
parser.add_argument("-n", "--nruns", help="The number of time to repeat the search", type=partial(greater, "number of runs", 0), default=10)
parser.add_argument("-c", "--cells", help="The number of cells to measure the coverage", type=partial(greater, "number of cells", 1), default=1000)
parser.add_argument("-q", "--quality", help="Enable quality", action="store_true")
args = parser.parse_args()


# choose parameters
POP_SIZE = args.population # -> 48 new individuals wil be evaluated each generation in order to match the nb of cores of MeSu beta with 2 nodes
NB_GEN = args.generation
OBJECT = args.object  # 'cuboid', 'mug.urdf', 'cylinder', 'deer.urdf', 'cylinder_r', 'glass.urdf'
ROBOT = args.robot  # 'baxter', 'pepper', 'kuka'
CONTROLLER = 'interpolate keypoints end pause grip'  # see controllers_dict for list
ALGO = 'ns_rand_multi_bd'  # algorithm
BD = 'pos_div_pos_grip'  # behavior descriptor type '2D', '3D', 'pos_div_grip', 'pos_div_pos_grip'
BOOTSTRAP_FOLDER = None
QUALITY = args.quality
AUTO_COLLIDE = True
NB_CELLS = args.cells; assert NB_CELLS>2  # number of cells for measurement
N_EXP = args.nruns



# for keypoints controllers
NB_KEYPOINTS = 3

if ROBOT == 'baxter':
    ENV_NAME = 'gym_baxter_grabbing:baxter_grasping-v0'
    GENE_PER_KEYPOINTS = 8  # baxter is joints space: 8 joints
    LINK_ID_CONTACT = [47, 48, 49, 50, 51, 52]  # link ids that can have a grasping contact
    NB_STEPS_TO_ROLLOUT = 10
    NB_ITER = int(6000 / NB_STEPS_TO_ROLLOUT)
    # set height thresh parameter
    if OBJECT == 'cuboid':
        HEIGHT_THRESH = -0.125
    elif OBJECT == 'cylinder':
        HEIGHT_THRESH = -0.16
    elif OBJECT == 'cylinder_r':
        HEIGHT_THRESH = -0.12
    elif OBJECT == 'mug.urdf':
        HEIGHT_THRESH = -0.15
    elif OBJECT == 'deer.urdf':
        HEIGHT_THRESH = -0.08
    elif OBJECT == 'glass.urdf':
        HEIGHT_THRESH = -0.11
    else: # otherwise, suppose the object is not that tall
        HEIGHT_THRESH = -0.15

if ROBOT == 'pepper':
    ENV_NAME = 'gym_baxter_grabbing:pepper_grasping-v0'
    GENE_PER_KEYPOINTS = 7  # pepper is controlled in joints space: 7 joints
    LINK_ID_CONTACT = list(range(36, 50))  # link ids that can have a grasping contact
    NB_STEPS_TO_ROLLOUT = 1
    NB_ITER = int(1000 / NB_STEPS_TO_ROLLOUT)
    # set height thresh parameter
    if OBJECT == 'cuboid':
        HEIGHT_THRESH = -0.15
    elif OBJECT == 'mug.urdf':
        HEIGHT_THRESH = -0.125
    elif OBJECT == 'deer.urdf':
        HEIGHT_THRESH = -0.095
    elif OBJECT == 'glass.urdf':
        HEIGHT_THRESH = -0.09

if ROBOT == 'kuka':
    ENV_NAME = 'gym_baxter_grabbing:kuka_grasping-v0'
    GENE_PER_KEYPOINTS = 9  # kuka is controlled in joints space: 7 joints
    LINK_ID_CONTACT = [8, 9, 10, 11, 12, 13]  # link ids that can have a grasping contact
    NB_STEPS_TO_ROLLOUT = 1
    NB_ITER = int(2500 / NB_STEPS_TO_ROLLOUT)
    # set height thresh parameter
    if OBJECT == 'cuboid':
        HEIGHT_THRESH = -0.08
    elif OBJECT == 'mug.urdf':
        HEIGHT_THRESH = -0.08
    elif OBJECT == 'deer.urdf':
        HEIGHT_THRESH = -0.03
    elif OBJECT == 'glass.urdf':
        HEIGHT_THRESH = -0.03

# for closed_loop control
if ROBOT == 'baxter':
    GENES = 344
# TODO: implement closed loop control for pepper and kuka

# choose minor parameters
PAUSE_FRAC = 0.66
MINI = True  # minimization problem (used for MAP-elites)
DISTANCE_THRESH = 0.6  # binary goal parameter
DIFF_OR_THRESH = 0.4  # threshold for clustering grasping orientations
COV_LIMIT = 0.1  # threshold for changing behavior descriptor in change_bd ns
N_LAG = int(200 / NB_STEPS_TO_ROLLOUT)  # number of steps before the grip time used in the pos_div_grip BD
ARCHIVE_LIMIT = 10000
N_REP_RAND = 4
SIGMA_RAND = 0.01
COUNT_SUCCESS = 0

if QUALITY:
    D_POS = utils.circle_coordinates(N_REP_RAND, SIGMA_RAND)

if ALGO == 'ns_rand_aurora':
    N_SAMPLES = 4


# if reset, create global env
# TODO: debug, for now RESET_MODE should be False
if RESET_MODE:
    ENV = gym.make(ENV_NAME, display=DISPLAY, obj=OBJECT, steps_to_roll=NB_STEPS_TO_ROLLOUT)


# choose diversity measure if gripping time is given by the controller
DIV_MEASURE = 'gripper orientation'  # 'gripper orientation', 'gripper orientation difference'

NOVELTY_METRIC = 'minkowski'

BD_INDEXES = None
MULTI_QUALITY_MEASURES = None
if BD == '2D':
    BD_BOUNDS = [[-0.35, 0.35], [-0.15, 0.2]]
if BD == '3D':
    BD_BOUNDS = [[-0.35, 0.35], [-0.15, 0.2], [-0.2, 0.5]]
if BD == 'pos_div_grip':
    BD_BOUNDS = [[-0.35, 0.35], [-0.15, 0.2], [-0.2, 0.5], [-1, 1], [-1, 1], [-1, 1], [-1, 1],
                 [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
    if ALGO == 'ns_rand_multi_bd':
        BD_INDEXES = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        NOVELTY_METRIC = ['minkowski', 'minkowski', 'minkowski']
        if QUALITY:
            # MULTI_QUALITY_MEASURES = [['mean positive slope', 'grasp robustness', None], ['min', 'min', None]]
            MULTI_QUALITY_MEASURES = [[None, 'grasp robustness', None], [None, 'max', None]]
if BD == 'pos_div_pos':
    BD_BOUNDS = [[-0.35, 0.35], [-0.15, 0.2], [-0.2, 0.5], [-1, 1], [-1, 1], [-1, 1], [-1, 1],
                 [-0.35, 0.35], [-0.15, 0.2], [-0.2, 0.5]]
    if ALGO == 'ns_rand_multi_bd':
        BD_INDEXES = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        NOVELTY_METRIC = ['minkowski', 'minkowski', 'minkowski']
        if QUALITY:
            # MULTI_QUALITY_MEASURES = [['mean positive slope', 'grasp robustness', None], ['min', 'min', None]]
            MULTI_QUALITY_MEASURES = [[None, 'grasp robustness', 'grasp robustness'], [None, 'max', 'max']]
if BD == 'pos_div_pos_grip':
    BD_BOUNDS = [[-0.35, 0.35], [-0.15, 0.2], [-0.2, 0.5], [-1, 1], [-1, 1], [-1, 1], [-1, 1],
                 [-0.35, 0.35], [-0.15, 0.2], [-0.2, 0.5], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
    if ALGO == 'ns_rand_multi_bd':
        BD_INDEXES = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
        NOVELTY_METRIC = ['minkowski', 'minkowski', 'minkowski', 'minkowski']
        if QUALITY:
            MULTI_QUALITY_MEASURES = [[None, 'grasp robustness', 'grasp robustness', None],
                                      [None, 'max', 'max', None]]
if BD == 'aurora':
    BD_BOUNDS = None
if ALGO == 'ns_rand_change_bd':
    BD = 'change_bd'
    # list of 3D bd and orientation bd
    BD_BOUNDS = [[[-0.35, 0.35], [-0.15, 0.2], [-0.2, 0.5]], [[-1, 1], [-1, 1], [-1, 1], [-1, 1]]]


# deal with Scoop parallelization
if PARALLELIZE:
    # container for behavior descriptor
    creator.create('BehaviorDescriptor', list)
    # container for extended behavior descriptor (only used for aurora)
    creator.create('ExtendedBehaviorDescriptor', list)
    # container for genetic info
    creator.create('GenInfo', dict)
    # container for info
    creator.create('Info', dict)
    # container for novelty
    if ALGO == 'ns_rand_multi_bd':
        # novelty must be a list, and selection is not used directly with DEAP
        creator.create('Novelty', list)
    else:
        creator.create('Novelty', base.Fitness, weights=(1.0,))
    # container for fitness
    if MINI:
        creator.create('Fit', base.Fitness, weights=(-1.0,))
    else:
        creator.create('Fit', base.Fitness, weights=(1.0,))

    # container for individual
    creator.create('Individual', list, behavior_descriptor=creator.BehaviorDescriptor,
                   novelty=creator.Novelty, fitness=creator.Fit, info=creator.Info,
                   gen_info=creator.GenInfo, extended_behavior_descriptor=creator.ExtendedBehaviorDescriptor)

    # set creator
    noveltysearch.set_creator(creator)


def diversity_measure(o):
    if DIV_MEASURE == 'gripper orientation difference':
        # object orientation at gripping time (quaternion)
        obj_or = o[1]

        # gripper orientation at gripping time (quaternion)
        grip_or = o[3]

        # pybullet is x, y, z, w whereas pyquaternion is w, x, y, z
        obj_or = pyq.Quaternion(obj_or[3], obj_or[0], obj_or[1], obj_or[2])
        grip_or = pyq.Quaternion(grip_or[3], grip_or[0], grip_or[1], grip_or[2])

        # difference:
        measure = obj_or.conjugate * grip_or

    if DIV_MEASURE == 'gripper orientation':
        grip_or = o[3]
        measure = pyq.Quaternion(grip_or[3], grip_or[0], grip_or[1], grip_or[2])

    return measure


def analyze_triumphants(triumphant_archive, run_name):
    if len(triumphant_archive) < 2:
        print('No individual completed the binary goal.')
        return None, None, None
    
    # analyze the triumphants following the diversity descriptor
    measure = 'diversity_descriptor'

    # sample the triumphant archive to reduce computational cost
    while len(triumphant_archive) >= 10000:
        triumphant_archive.pop(random.randint(0, len(triumphant_archive) - 1))

    random.shuffle(triumphant_archive)
    
    nb_of_triumphants = len(triumphant_archive)

    # compute coverage and uniformity metrics: easy approach, use CVT cells in quaternion space
    bounds = [[-1, 1], [-1, 1], [-1, 1], [-1, 1]]
    cvt = utils.CVT(NB_CELLS, bounds)
    grid = np.zeros((NB_CELLS,))
    for ind in triumphant_archive:
        or_diff = ind.info.values[measure]
        or_diff_arr = [or_diff[0], or_diff[1], or_diff[2], or_diff[3]]
        grid_index = cvt.get_grid_index(or_diff_arr)
        grid[grid_index] += 1
    coverage = np.count_nonzero(grid) / NB_CELLS
    uniformity = utils.compute_uniformity(grid)

    # cluster the triumphants with respect to grasping descriptor
    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', compute_full_tree=True,
                                         distance_threshold=DIFF_OR_THRESH, linkage='average')
    # compute distance matrix
    X = np.zeros((nb_of_triumphants, nb_of_triumphants))
    for x in range(nb_of_triumphants):
        for y in range(nb_of_triumphants):
            if x == y:
                X[x, y] = 0
            else:
                triumphant_a = triumphant_archive[x].info.values[measure]
                triumphant_b = triumphant_archive[y].info.values[measure]
                X[x, y] = pyq.Quaternion.absolute_distance(triumphant_a, triumphant_b)

    # fit distance matrix
    clustering.fit(X)

    number_of_clusters = clustering.n_clusters_
    labels = clustering.labels_

    clustered_triumphants = []
    for i in range(number_of_clusters):
        members_of_cluster = []
        for j, triumphant in enumerate(triumphant_archive):
            if labels[j] == i:
                members_of_cluster.append(triumphant)
        clustered_triumphants.append(members_of_cluster)

    print(number_of_clusters, 'types of grasping were found.')
    print('Coverage of', coverage, 'and uniformity of', uniformity)

    # saving the triumphants
    for i in range(len(clustered_triumphants)):
        # save first 3 grasping of each types
        for j in range(3):
            if len(clustered_triumphants[i]) > j:
                ind = np.around(np.array(clustered_triumphants[i][j]), 3)

                np.save(run_name + 'type' + str(i) + '_' + str(j), ind,
                        allow_pickle=True)

    return coverage, uniformity, clustered_triumphants
    

def two_d_bd(individual):
    """Evaluates an individual: computes its value in the behavior descriptor space,
    and its fitness value.
    In this case, we consider the behavior space as the end position of the object in
    the 2D plane (no grasping needed, only pushing).

    Args:
        individual (Individual): an individual

    Returns:
        tuple: tuple of behavior (list) and fitness(tuple)
    """
    if RESET_MODE:
        global ENV
        ENV.reset()
    else:
        ENV = gym.make(ENV_NAME, display=DISPLAY, obj=OBJECT, steps_to_roll=NB_STEPS_TO_ROLLOUT)

    individual = np.around(np.array(individual), 3)
    # initialize controller
    controller_info = controllers_info_dict[CONTROLLER]
    controller = controllers_dict[CONTROLLER](individual, controller_info, initial=ENV.get_action())
    action = controller.initial_action

    for i in range(NB_ITER):
        ENV.render()
        # apply previously chosen action
        o, r, eo, info = ENV.step(action)

        if i == 0:
            initial_object_position = o[0]
        if controller.open_loop:
            action = controller.get_action(i)
        else:
            action = controller.get_action(i, o)

        if eo:
            break
    # use last info to compute behavior and fitness
    behavior = [o[0][0] - initial_object_position[0], o[0][1] - initial_object_position[1]]  # last position of object

    # bound behavior descriptor on table
    utils.bound(behavior, BD_BOUNDS)

    # compute fitness
    fitness = utils.list_l2_norm(behavior, [0, 0])

    info = {}
    if not RESET_MODE:
        ENV.close()

    return (behavior, (fitness,), info)


def three_d_bd(individual):
    """Evaluates an individual: computes its value in the behavior descriptor space,
    and its fitness value.
    In this case, we consider the behavior space as the end position of the object in
    the 3D volume.

    Args:
        individual (Individual): an individual

    Returns:
        tuple: tuple of behavior (list) and fitness(tuple)
    """
    if RESET_MODE:
        global ENV
        ENV.reset()
    else:
        ENV = gym.make(ENV_NAME, display=DISPLAY, obj=OBJECT, steps_to_roll=NB_STEPS_TO_ROLLOUT)

    individual = np.around(np.array(individual), 3)
    
    # initialize controller
    controller_info = controllers_info_dict[CONTROLLER]
    controller = controllers_dict[CONTROLLER](individual, controller_info, initial=ENV.get_action())
    action = controller.initial_action

    # for precise measure when we have the gripper assumption
    grabbed = False

    info = {}

    for i in range(NB_ITER):
        ENV.render()
        # apply previously chosen action
        o, r, eo, inf = ENV.step(action)

        # choose action
        if controller.open_loop:
            action = controller.get_action(i)
        else:
            action = controller.get_action(i, o)

        if i == 0:
            initial_object_position = o[0]

        if eo:
            break
        
        if hasattr(controller, 'grip_time'):
            # we are in the case where the gripping time is given
            # in consequence, we can do the precise measure of the grabbing orientation
            if action[-1] == -1 and not grabbed:
                # first action that orders the grabbing

                measure_grip_time = diversity_measure(o)
                grabbed = True

    # use last info to compute behavior and fitness
    behavior = [o[0][0] - initial_object_position[0], o[0][1] - initial_object_position[1],
                o[0][2]]  # last position of object

    # bound behavior descriptor on table
    if ALGO == 'ns_rand_change_bd':
        utils.bound(behavior, BD_BOUNDS[0])
    else:
        utils.bound(behavior, BD_BOUNDS)

    # compute fitness
    fitness = behavior[2]

    # choose if individual satisfied the binary goal
    dist = utils.list_l2_norm(o[0], o[2])
    binary_goal = False
    if o[0][2] > HEIGHT_THRESH and dist < DISTANCE_THRESH:
        binary_goal = True
    info['binary goal'] = binary_goal

    if binary_goal:
        if hasattr(controller, 'grip_time'):
            info['diversity_descriptor'] = measure_grip_time
        else:
            # last object orientation (quaternion)
            obj_or = o[1]

            # last gripper orientation (quaternion)
            grip_or = o[3]

            # pybullet is x, y, z, w whereas pyquaternion is w, x, y, z
            obj_or = pyq.Quaternion(obj_or[3], obj_or[0], obj_or[1], obj_or[2])
            grip_or = pyq.Quaternion(grip_or[3], grip_or[0], grip_or[1], grip_or[2])

            # difference:
            diff_or = obj_or.conjugate * grip_or
            info['diversity_descriptor'] = diff_or
    if not RESET_MODE:
        ENV.close()

    return (behavior, (fitness,), info)


def pos_div_grip_bd(individual):
    """Evaluates an individual: computes its value in the behavior descriptor space,
    and its fitness value.
    In this case, we consider the behavior space where we give the maximum amount of information
    as possible to the algorithm to establish a strong baseline.

    controller with grip_time is required

    3 descriptors:
    -the end position of the object in the 3D volume, always eligible
    -the measure described by diversity_measure, eligible if the object is grabbed
    -the orientation of the gripper N_LAG steps before the gripping time, always eligible

    Args:
        individual (Individual): an individual

    Returns:
        tuple: tuple of behavior (list) and fitness(tuple)
    """
    if RESET_MODE:
        global ENV
        ENV.reset()
    else:
        ENV = gym.make(ENV_NAME, display=DISPLAY, obj=OBJECT, steps_to_roll=NB_STEPS_TO_ROLLOUT)

    global COUNT_SUCCESS

    individual = np.around(np.array(individual), 3)

    # initialize controller
    controller_info = controllers_info_dict[CONTROLLER]
    controller = controllers_dict[CONTROLLER](individual, controller_info, initial=ENV.get_action())
    assert(hasattr(controller, 'grip_time'))
    # lag_time = controller.grip_time - N_LAG
    lag_time = NB_ITER / 2
    action = controller.initial_action

    # monitor auto-collision
    auto_collision = False

    # for precise measure when we have the gripper assumption
    already_touched = False
    already_grasped = False
    grasped_before_touch = False

    # for measure at lag time
    lag_measured = False

    measure_grip_time = None

    touch_idx = []

    # to compute quality for B1
    positive_dist_slope = 0
    prev_dist = None

    info = {}

    if ALGO == 'map_elites':
        # define energy criterion
        energy = 0

    for i in range(NB_ITER):
        ENV.render()
        # apply previously chosen action
        o, r, eo, inf = ENV.step(action)

        prev_action = action

        # choose action
        if controller.open_loop:
            action = controller.get_action(i)
        else:
            action = controller.get_action(i, o)

        if i == 0:
            initial_object_position = o[0]

        if eo:
            break
        
        if i >= controller.grip_time and not already_grasped:
            # first action that orders the gripper closure
            # measure_grip_time = diversity_measure(o)
            already_grasped = True

        touch = len(inf['contact_points']) > 0
        touch_id = 0
        if touch:
            touch_id = inf['contact_points'][0][3]
            touch_idx.append(touch_id)
        relevant_touch = touch and (touch_id in LINK_ID_CONTACT)
        if relevant_touch and not already_touched:
            # first touch of object
            measure_grip_time = diversity_measure(o)
            already_touched = True
            if already_grasped:
                grasped_before_touch = True
        
        if i >= lag_time and not lag_measured:
            # gripper orientation
            grip_or_lag = o[3]
            lag_measured = True

        # quality 1 measured during the whole trajectory
        if QUALITY:
            # only done one step after the start
            if prev_dist is None:
                # distance between gripper and object
                prev_dist = utils.list_l2_norm(o[0], o[2])
            else:
                new_dist = utils.list_l2_norm(o[0], o[2])
                differential_dist = new_dist - prev_dist
                if differential_dist > 0:
                    positive_dist_slope += differential_dist
            
                prev_dist = new_dist

        # if robot has a self-collision monitoring
        if 'self contact_points' in inf and AUTO_COLLIDE:
            if len(inf['self contact_points']) != 0:
                auto_collision = True
                break

        if ALGO == 'map_elites':
            energy += utils.list_l2_norm(action, prev_action) ** 2
    
    if auto_collision:
        behavior = [None, None, None, None, None, None, None, None, None, None, None]
        if ALGO != 'ns_rand_multi_bd':
            behavior = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        fitness = -float('inf')
        info = {'binary goal': False, 'auto_collided': True}
        if not RESET_MODE:
            ENV.close()
        return (behavior, (fitness,), info)

    # use last info to compute behavior and fitness
    behavior = [o[0][0] - initial_object_position[0], o[0][1] - initial_object_position[1],
                o[0][2]]  # last position of object

    utils.bound(behavior, BD_BOUNDS[0:3])

    # compute fitness
    if ALGO == 'map_elites':
        fitness = energy
    else:
        fitness = behavior[2]

    # append 4 times None to behavior in case no grasping (modified later)
    for _ in range(4):
        behavior.append(None)

    # choose if individual satisfied the binary goal
    dist = utils.list_l2_norm(o[0], o[2])
    binary_goal = False
    if o[0][2] > HEIGHT_THRESH and dist < DISTANCE_THRESH:
        binary_goal = True
    info['binary goal'] = binary_goal

    if binary_goal:
        COUNT_SUCCESS += 1
        if measure_grip_time is None:
            # print('Individual grasped without touching any contact links')
            info['binary goal'] = False
        else:
            info['diversity_descriptor'] = measure_grip_time
            behavior[3] = measure_grip_time[0]  # Quat to array
            behavior[4] = measure_grip_time[1]
            behavior[5] = measure_grip_time[2]
            behavior[6] = measure_grip_time[3]
    
    # BD 3 only active if trajectory touched the object
    if already_touched:
        behavior.append(grip_or_lag[3])
        behavior.append(grip_or_lag[0])
        behavior.append(grip_or_lag[1])
        behavior.append(grip_or_lag[2])
    else:
        for _ in range(4):
            behavior.append(None)

    if not RESET_MODE:
        ENV.close()

    if ALGO != 'ns_rand_multi_bd':
        for i, b in enumerate(behavior):
            if b is None:
                behavior[i] = 0
        return (behavior, (fitness,), info)

    if QUALITY:
        if grasped_before_touch:
            # penalize more
            info['mean positive slope'] = 4 * positive_dist_slope / NB_ITER
        elif already_touched:
            info['mean positive slope'] = positive_dist_slope / NB_ITER
            if (not inf['closed gripper']) and relevant_touch:
                # gripper is not entirely closed at the end, and is touching the object
                info['mean positive slope'] -= 1
        else:
            info['mean positive slope'] = positive_dist_slope / NB_ITER + 1

    if QUALITY and binary_goal:
        # re-evaluate with random initial positions to assess robustness as quality
        reference = [o[0][0], o[0][1], o[0][2]]
        last_pos_obj = []
        count = 0
        for rep in range(N_REP_RAND):
            if RESET_MODE:
                ENV.reset(delta_pos=D_POS[rep])
            else:
                ENV = gym.make(ENV_NAME, display=DISPLAY, obj=OBJECT, delta_pos=D_POS[rep],
                           steps_to_roll=NB_STEPS_TO_ROLLOUT)

            # initialize controller
            controller_info = controllers_info_dict[CONTROLLER]
            controller = controllers_dict[CONTROLLER](individual, controller_info, initial=ENV.get_action())
            action = controller.initial_action

            for i in range(NB_ITER):
                ENV.render()
                # apply previously chosen action
                o, r, eo, inf = ENV.step(action)

                # choose action
                if controller.open_loop:
                    action = controller.get_action(i)
                else:
                    action = controller.get_action(i, o)

                if i == 0:
                    initial_object_position = o[0]

                if eo:
                    break
            dist = utils.list_l2_norm(o[0], o[2])
            binary_goal = False
            if o[0][2] > HEIGHT_THRESH and dist < DISTANCE_THRESH:
                binary_goal = True
            
            if binary_goal:
                count += 1
                last_pos_obj.append([o[0][0], o[0][1], o[0][2]])

            if not RESET_MODE:
                ENV.close()

        mean_dist = 0
        for last_pos in last_pos_obj:
            mean_dist += utils.list_l2_norm(reference, last_pos)
        if count != 0:
            mean_dist = mean_dist / count

        grasp_rob = count + 1 / (1 + 0.00000001 + mean_dist)
        info['grasp robustness'] = grasp_rob
    
    return (behavior, (fitness,), info)


def pos_div_pos_bd(individual):

    if RESET_MODE:
        global ENV
        ENV.reset()
    else:
        ENV = gym.make(ENV_NAME, display=DISPLAY, obj=OBJECT, steps_to_roll=NB_STEPS_TO_ROLLOUT)

    global COUNT_SUCCESS

    individual = np.around(np.array(individual), 3)

    # initialize controller
    controller_info = controllers_info_dict[CONTROLLER]
    controller = controllers_dict[CONTROLLER](individual, controller_info, initial=ENV.get_action())
    assert(hasattr(controller, 'grip_time'))

    action = controller.initial_action

    # monitor auto-collision
    auto_collision = False

    # for precise measure when we have the gripper assumption
    already_touched = False
    already_grasped = False
    grasped_before_touch = False

    measure_grip_time = None
    pos_touch_time = None

    touch_idx = []

    # to compute quality for B1
    positive_dist_slope = 0
    prev_dist = None

    info = {}

    if ALGO == 'map_elites':
        # define energy criterion
        energy = 0

    for i in range(NB_ITER):
        ENV.render()
        # apply previously chosen action
        o, r, eo, inf = ENV.step(action)

        prev_action = action

        # choose action
        if controller.open_loop:
            action = controller.get_action(i)
        else:
            action = controller.get_action(i, o)

        if i == 0:
            initial_object_position = o[0]

        if eo:
            break
        
        if i >= controller.grip_time and not already_grasped:
            # first action that orders the gripper closure
            # measure_grip_time = diversity_measure(o)
            already_grasped = True

        touch = len(inf['contact_points']) > 0
        touch_id = 0
        if touch:
            touch_id = inf['contact_points'][0][3]
            touch_idx.append(touch_id)
        relevant_touch = touch and (touch_id in LINK_ID_CONTACT)
        if relevant_touch and not already_touched:
            # first touch of object
            measure_grip_time = diversity_measure(o)
            pos_touch_time = o[2]
            already_touched = True
            if already_grasped:
                grasped_before_touch = True

        # quality 1 measured during the whole trajectory
        if QUALITY:
            # only done one step after the start
            if prev_dist is None:
                # distance between gripper and object
                prev_dist = utils.list_l2_norm(o[0], o[2])
            else:
                new_dist = utils.list_l2_norm(o[0], o[2])
                differential_dist = new_dist - prev_dist
                if differential_dist > 0:
                    positive_dist_slope += differential_dist
            
                prev_dist = new_dist

        # if robot has a self-collision monitoring
        if 'self contact_points' in inf and AUTO_COLLIDE:
            if len(inf['self contact_points']) != 0:
                auto_collision = True
                break

        if ALGO == 'map_elites':
            energy += utils.list_l2_norm(action, prev_action) ** 2
    
    if auto_collision:
        behavior = [None]*10#[None, None, None, None, None, None, None, None, None, None, None, None, None, None]
        if ALGO != 'ns_rand_multi_bd':
            behavior = [0]*10#[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        fitness = -float('inf')
        info = {'binary goal': False, 'auto_collided': True}
        if not RESET_MODE:
            ENV.close()
        return (behavior, (fitness,), info)

    # use last info to compute behavior and fitness
    behavior = [o[0][0] - initial_object_position[0], o[0][1] - initial_object_position[1],
                o[0][2]]  # last position of object

    utils.bound(behavior, BD_BOUNDS[0:3])

    # compute fitness
    if ALGO == 'map_elites':
        fitness = energy
    else:
        fitness = behavior[2]

    # append 4 times None to behavior in case no grasping (modified later)
    for _ in range(7):
        behavior.append(None)

    # choose if individual satisfied the binary goal
    dist = utils.list_l2_norm(o[0], o[2])
    binary_goal = False
    if o[0][2] > HEIGHT_THRESH and dist < DISTANCE_THRESH:
        binary_goal = True
    info['binary goal'] = binary_goal

    if binary_goal:
        COUNT_SUCCESS += 1
        if measure_grip_time is None:
            # print('Individual grasped without touching any contact links')
            info['binary goal'] = False
        else:
            info['diversity_descriptor'] = measure_grip_time
            behavior[3] = measure_grip_time[0]  # Quat to array
            behavior[4] = measure_grip_time[1]
            behavior[5] = measure_grip_time[2]
            behavior[6] = measure_grip_time[3]
            behavior[7] = pos_touch_time[0]
            behavior[8] = pos_touch_time[1]
            behavior[9] = pos_touch_time[2]

    if not RESET_MODE:
        ENV.close()

    if ALGO != 'ns_rand_multi_bd':
        for i, b in enumerate(behavior):
            if b is None:
                behavior[i] = 0
        return (behavior, (fitness,), info)

    if QUALITY:
        if grasped_before_touch:
            # penalize more
            info['mean positive slope'] = 4 * positive_dist_slope / NB_ITER
        elif already_touched:
            info['mean positive slope'] = positive_dist_slope / NB_ITER
            if (not inf['closed gripper']) and relevant_touch:
                # gripper is not entirely closed at the end, and is touching the object
                info['mean positive slope'] -= 1
        else:
            info['mean positive slope'] = positive_dist_slope / NB_ITER + 1

    if QUALITY and binary_goal:
        # re-evaluate with random initial positions to assess robustness as quality
        reference = [o[0][0], o[0][1], o[0][2]]
        last_pos_obj = []
        count = 0
        for rep in range(N_REP_RAND):
            if RESET_MODE:
                ENV.reset(delta_pos=D_POS[rep])
            else:
                ENV = gym.make(ENV_NAME, display=DISPLAY, obj=OBJECT, delta_pos=D_POS[rep],
                           steps_to_roll=NB_STEPS_TO_ROLLOUT)

            # initialize controller
            controller_info = controllers_info_dict[CONTROLLER]
            controller = controllers_dict[CONTROLLER](individual, controller_info, initial=ENV.get_action())
            action = controller.initial_action

            for i in range(NB_ITER):
                ENV.render()
                # apply previously chosen action
                o, r, eo, inf = ENV.step(action)

                # choose action
                if controller.open_loop:
                    action = controller.get_action(i)
                else:
                    action = controller.get_action(i, o)

                if i == 0:
                    initial_object_position = o[0]

                if eo:
                    break
            dist = utils.list_l2_norm(o[0], o[2])
            binary_goal = False
            if o[0][2] > HEIGHT_THRESH and dist < DISTANCE_THRESH:
                binary_goal = True
            
            if binary_goal:
                count += 1
                last_pos_obj.append([o[0][0], o[0][1], o[0][2]])

            if not RESET_MODE:
                ENV.close()

        mean_dist = 0
        for last_pos in last_pos_obj:
            mean_dist += utils.list_l2_norm(reference, last_pos)
        if count != 0:
            mean_dist = mean_dist / count

        grasp_rob = count + 1 / (1 + 0.00000001 + mean_dist)
        info['grasp robustness'] = grasp_rob
    
    return (behavior, (fitness,), info)


def pos_div_pos_grip_bd(individual):

    if RESET_MODE:
        global ENV
        ENV.reset()
    else:
        ENV = gym.make(ENV_NAME, display=DISPLAY, obj=OBJECT, steps_to_roll=NB_STEPS_TO_ROLLOUT)

    global COUNT_SUCCESS

    individual = np.around(np.array(individual), 3)

    # initialize controller
    controller_info = controllers_info_dict[CONTROLLER]
    controller = controllers_dict[CONTROLLER](individual, controller_info, initial=ENV.get_action())
    assert(hasattr(controller, 'grip_time'))
    # lag_time = controller.grip_time - N_LAG
    lag_time = NB_ITER / 2
    action = controller.initial_action

    # monitor auto-collision
    auto_collision = False

    # for precise measure when we have the gripper assumption
    already_touched = False
    already_grasped = False
    grasped_before_touch = False
    closing = None

    # for measure at lag time
    lag_measured = False

    measure_grip_time = None
    pos_touch_time = None

    touch_idx = []

    # to compute quality for B1
    positive_dist_slope = 0
    prev_dist = None

    info, grip_info = {}, {}

    if ALGO == 'map_elites':
        # define energy criterion
        energy = 0

    for i in range(NB_ITER):
        ENV.render()
        # apply previously chosen action
        o, r, eo, inf = ENV.step(action)

        prev_action = action

        # choose action
        if controller.open_loop:
            action = controller.get_action(i)
        else:
            action = controller.get_action(i, o)

        if i == 0:
            initial_object_position = o[0]

        if eo:
            break
        
        if i >= controller.grip_time and not already_grasped:
            # first action that orders the gripper closure
            # measure_grip_time = diversity_measure(o)
            already_grasped = True
            grip_info["contact object table"] = inf["contact object table"] # the object should touch the table while grasping
            closing = i # the object should be grasped right after closing the gripper

        touch = len(inf['contact_points']) > 0
        touch_id = 0
        if touch:
            touch_id = inf['contact_points'][0][3]
            touch_idx.append(touch_id)
        relevant_touch = touch and (touch_id in LINK_ID_CONTACT)
        if relevant_touch and not already_touched:
            # first touch of object
            measure_grip_time = diversity_measure(o)
            pos_touch_time = o[2]
            already_touched = True
            if already_grasped:
                grasped_before_touch = True
        if relevant_touch and closing: # get the time step difference between when start closing and touching
            grip_info["time close touch"] = i - closing
            closing = None
        
        if i >= lag_time and not lag_measured:
            # gripper orientation
            grip_or_lag = o[3]
            lag_measured = True

        # quality 1 measured during the whole trajectory
        if QUALITY:
            # only done one step after the start
            if prev_dist is None:
                # distance between gripper and object
                prev_dist = utils.list_l2_norm(o[0], o[2])
            else:
                new_dist = utils.list_l2_norm(o[0], o[2])
                differential_dist = new_dist - prev_dist
                if differential_dist > 0:
                    positive_dist_slope += differential_dist
            
                prev_dist = new_dist

        # if robot has a self-collision monitoring
        if 'self contact_points' in inf and AUTO_COLLIDE:
            if len(inf['self contact_points']) != 0:
                #print("collision", i, inf['self contact_points'][0][3], inf['self contact_points'][0][4])
                auto_collision = True
                break

        if ALGO == 'map_elites':
            energy += utils.list_l2_norm(action, prev_action) ** 2
    
    if auto_collision:
        behavior = [None]*14#[None, None, None, None, None, None, None, None, None, None, None, None, None, None]
        if ALGO != 'ns_rand_multi_bd':
            behavior = [0]*14#[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        fitness = -float('inf')
        info = {'binary goal': False, 'auto_collided': True}
        if not RESET_MODE:
            ENV.close()
        return (behavior, (fitness,), info)

    # use last info to compute behavior and fitness
    behavior = [o[0][0] - initial_object_position[0], o[0][1] - initial_object_position[1],
                o[0][2]]  # last position of object

    utils.bound(behavior, BD_BOUNDS[0:3])

    # compute fitness
    if ALGO == 'map_elites':
        fitness = energy
    else:
        fitness = behavior[2]

    # append 4 times None to behavior in case no grasping (modified later)
    for _ in range(7):
        behavior.append(None)

    # choose if individual satisfied the binary goal
    dist = utils.list_l2_norm(o[0], o[2])
    binary_goal = False
    
    #relevant_contact = [c for c in inf['contact_points'] if c[3] in LINK_ID_CONTACT] # contact with the gripper
    # the object should not touch the table neither the plane, must touch the gripper without penetration (with a margin of 0.005), be grasped right after closing the gripper (within 1s), touch the table when the gripper is closing
    #if len(inf['contact object plane']+(inf['contact object table'] if 'contact object table' in inf.keys() else []))==0 and len(relevant_contact)>0 and np.all([c[8]>-0.01 for c in inf['contact_points']]) and grip_info['time close touch']<1*240/NB_STEPS_TO_ROLLOUT:# and len(grip_info['contact object table'])>0:
    if o[0][2] > HEIGHT_THRESH and dist < DISTANCE_THRESH:
        binary_goal = True
    info['binary goal'] = binary_goal

    if binary_goal:
        COUNT_SUCCESS += 1
        if measure_grip_time is None:
            # print('Individual grasped without touching any contact links')
            info['binary goal'] = False
        else:
            info['diversity_descriptor'] = measure_grip_time
            behavior[3] = measure_grip_time[0]  # Quat to array
            behavior[4] = measure_grip_time[1]
            behavior[5] = measure_grip_time[2]
            behavior[6] = measure_grip_time[3]
            behavior[7] = pos_touch_time[0]
            behavior[8] = pos_touch_time[1]
            behavior[9] = pos_touch_time[2]

    # BD 3 only active if trajectory touched the object
    if already_touched:
        behavior.append(grip_or_lag[3])
        behavior.append(grip_or_lag[0])
        behavior.append(grip_or_lag[1])
        behavior.append(grip_or_lag[2])
    else:
        for _ in range(4):
            behavior.append(None)

    if not RESET_MODE:
        ENV.close()

    if ALGO != 'ns_rand_multi_bd':
        for i, b in enumerate(behavior):
            if b is None:
                behavior[i] = 0
        return (behavior, (fitness,), info)

    if QUALITY:
        if grasped_before_touch:
            # penalize more
            info['mean positive slope'] = 4 * positive_dist_slope / NB_ITER
        elif already_touched:
            info['mean positive slope'] = positive_dist_slope / NB_ITER
            if (not inf['closed gripper']) and relevant_touch:
                # gripper is not entirely closed at the end, and is touching the object
                info['mean positive slope'] -= 1
        else:
            info['mean positive slope'] = positive_dist_slope / NB_ITER + 1

    if QUALITY and binary_goal:
        # re-evaluate with random initial positions to assess robustness as quality
        reference = [o[0][0], o[0][1], o[0][2]]
        last_pos_obj = []
        count = 0
        for rep in range(N_REP_RAND):
            if RESET_MODE:
                ENV.reset(delta_pos=D_POS[rep])
            else:
                ENV = gym.make(ENV_NAME, display=DISPLAY, obj=OBJECT, delta_pos=D_POS[rep],
                           steps_to_roll=NB_STEPS_TO_ROLLOUT)

            # initialize controller
            controller_info = controllers_info_dict[CONTROLLER]
            controller = controllers_dict[CONTROLLER](individual, controller_info, initial=ENV.get_action())
            action = controller.initial_action

            for i in range(NB_ITER):
                ENV.render()
                # apply previously chosen action
                o, r, eo, inf = ENV.step(action)

                # choose action
                if controller.open_loop:
                    action = controller.get_action(i)
                else:
                    action = controller.get_action(i, o)

                if i == 0:
                    initial_object_position = o[0]

                if eo:
                    break
            dist = utils.list_l2_norm(o[0], o[2])
            binary_goal = False
            if o[0][2] > HEIGHT_THRESH and dist < DISTANCE_THRESH:
                binary_goal = True
            
            if binary_goal:
                count += 1
                last_pos_obj.append([o[0][0], o[0][1], o[0][2]])

            if not RESET_MODE:
                ENV.close()

        mean_dist = 0
        for last_pos in last_pos_obj:
            mean_dist += utils.list_l2_norm(reference, last_pos)
        if count != 0:
            mean_dist = mean_dist / count

        grasp_rob = count + 1 / (1 + 0.00000001 + mean_dist)
        info['grasp robustness'] = grasp_rob
    
    return (behavior, (fitness,), info)


def eval_sucessfull_ind(individual, obstacle_pos=None, obstacle_size=None):

    if obstacle_pos is None:
        ENV = gym.make(ENV_NAME, display=DISPLAY, obj=OBJECT, steps_to_roll=NB_STEPS_TO_ROLLOUT)
    else:
        ENV = gym.make(ENV_NAME, display=DISPLAY, obj=OBJECT, steps_to_roll=NB_STEPS_TO_ROLLOUT,
                       obstacle=True, obstacle_pos=obstacle_pos, obstacle_size=obstacle_size)

    individual = np.around(np.array(individual), 3)

    # initialize controller
    controller_info = controllers_info_dict[CONTROLLER]
    controller = controllers_dict[CONTROLLER](individual, controller_info, initial=ENV.get_action())
    action = controller.initial_action

    # for precise measure when we have the gripper assumption
    already_grasped = False

    count_keypoint = 0
    closed_keypoint_idx = 0
    traj_array = []
    closed = False

    for i in range(NB_ITER):
        ENV.render()
        # apply previously chosen action
        o, r, eo, inf = ENV.step(action)

        # choose action
        if controller.open_loop:
            action = controller.get_action(i)
        else:
            action = controller.get_action(i, o)

        if i % 15 == 0 and SAVE_TRAJ:
            count_keypoint += 1
            joint_config = o[4][10:17]
            traj_array.append(joint_config)
            if not closed and already_grasped:
                closed = True
                closed_keypoint_idx = count_keypoint

        if eo:
            break
        
        if i >= controller.grip_time and not already_grasped:
            # first action that orders the gripper closure
            already_grasped = True

        # if robot has a self-collision monitoring
        if 'self contact_points' in inf and AUTO_COLLIDE:
            if len(inf['self contact_points']) != 0:
                raise Exception('Auto collision detected')

    # choose if individual satisfied the binary goal
    dist = utils.list_l2_norm(o[0], o[2])
    binary_goal = False
    if o[0][2] > HEIGHT_THRESH and dist < DISTANCE_THRESH:
        binary_goal = True

    if not RESET_MODE:
        ENV.close()

    if SAVE_TRAJ:
        traj_array.append(closed_keypoint_idx)

    return binary_goal, traj_array


def aurora_bd(individual):

    if RESET_MODE:
        global ENV
        ENV.reset()
    else:
        ENV = gym.make(ENV_NAME, display=DISPLAY, obj=OBJECT, steps_to_roll=NB_STEPS_TO_ROLLOUT)

    individual = np.around(np.array(individual), 3)

    # initialize controller
    controller_info = controllers_info_dict[CONTROLLER]
    controller = controllers_dict[CONTROLLER](individual, controller_info, initial=ENV.get_action())

    action = controller.initial_action

    # for precise measure when we have the gripper assumption
    already_touched = False

    measure_grip_time = None

    touch_idx = []

    info = {}
    sample_points = []
    sample_step = NB_ITER / (N_SAMPLES + 1)
    for point in range(N_SAMPLES):
        sample_points.append(sample_step * (point + 1))
    
    behavior = []

    for i in range(NB_ITER):
        ENV.render()
        # apply previously chosen action
        o, r, eo, inf = ENV.step(action)

        # choose action
        if controller.open_loop:
            action = controller.get_action(i)
        else:
            action = controller.get_action(i, o)

        if eo:
            break

        touch = len(inf['contact_points']) > 0
        touch_id = 0
        if touch:
            touch_id = inf['contact_points'][0][3]
            touch_idx.append(touch_id)
        relevant_touch = touch and (touch_id in LINK_ID_CONTACT)
        if relevant_touch and not already_touched:
            # first touch of object
            measure_grip_time = diversity_measure(o)
            already_touched = True

        if i in sample_points:
            # we sample the trajectory to feed the high dimensional BD
            flattened_obs = [item for sublist in o for item in sublist]
            obs_bounds = [[-1, 1]] * 14 + [[-math.pi, math.pi]] * len(o[4])
            utils.bound(flattened_obs, obs_bounds)
            utils.normalize(flattened_obs, obs_bounds)
            behavior.append(flattened_obs)

    # compute fitness
    fitness = o[0][2]

    # choose if individual satisfied the binary goal
    dist = utils.list_l2_norm(o[0], o[2])
    binary_goal = False
    if o[0][2] > HEIGHT_THRESH and dist < DISTANCE_THRESH:
        binary_goal = True
    info['binary goal'] = binary_goal

    if binary_goal:
        if measure_grip_time is None:
            # print('Individual grasped without touching any contact links')
            info['binary goal'] = False
        else:
            info['diversity_descriptor'] = measure_grip_time

    if not RESET_MODE:
        ENV.close()

    behavior_flat = [item for sublist in behavior for item in sublist]

    return (behavior_flat, (fitness,), info)


def choose_evaluation_function(info_change_bd):
    """In case of multi_ns bd, choose the evaluation function based on the dict sent by the ns algorithm.

    Args:
        info_change_bd (dict): dictionnary containing the info relevant to choose the bd

    Returns:
        int : index of the evaluation function and bd_bounds
    """
    index = 0

    if not info_change_bd['changed']:
        # bd has not been changed yet
        if info_change_bd['coverage'] >= COV_LIMIT:
            index = 1
    else:
        index = 1

    return index


controllers_dict = {'discrete keypoints': controllers.DiscreteKeyPoints,
                    'interpolate keypoints': controllers.InterpolateKeyPoints,
                    'interpolate keypoints end pause': controllers.InterpolateKeyPointsEndPause,
                    'interpolate keypoints end pause grip': controllers.InterpolateKeyPointsEndPauseGripAssumption,
                    'closed loop end pause grip': controllers.ClosedLoopEndPauseGripAssumption}
controllers_info_dict = {'interpolate keypoints end pause grip': {'pause_frac': PAUSE_FRAC, 'n_iter': NB_ITER,
                                                                  'NB_KEYPOINTS': NB_KEYPOINTS,
                                                                  'GENE_PER_KEYPOINTS': GENE_PER_KEYPOINTS},
                         'interpolate keypoints end pause': {'pause_frac': PAUSE_FRAC, 'n_iter': NB_ITER,
                                                             'NB_KEYPOINTS': NB_KEYPOINTS,
                                                             'GENE_PER_KEYPOINTS': GENE_PER_KEYPOINTS},
                         'discrete keypoints': {'n_iter': NB_ITER,
                                                'NB_KEYPOINTS': NB_KEYPOINTS,
                                                'GENE_PER_KEYPOINTS': GENE_PER_KEYPOINTS},
                         'interpolate keypoints': {'n_iter': NB_ITER,
                                                   'NB_KEYPOINTS': NB_KEYPOINTS,
                                                   'GENE_PER_KEYPOINTS': GENE_PER_KEYPOINTS},
                         'closed loop end pause grip': {'n_iter': NB_ITER, 'pause_frac': PAUSE_FRAC}}
bd_dict = {'2D': two_d_bd,
           '3D': three_d_bd,
           'pos_div_grip': pos_div_grip_bd,
           'pos_div_pos_grip': pos_div_pos_grip_bd,
           'pos_div_pos': pos_div_pos_bd,
           'aurora': aurora_bd}

if __name__ == "__main__":
    print(f"pop size={POP_SIZE}, ngen={NB_GEN}, object={OBJECT}, robot={ROBOT}, quality={QUALITY}, autocollide={AUTO_COLLIDE}, nexp={N_EXP}, reset mode={RESET_MODE}, parallelize={PARALLELIZE}, controller={CONTROLLER}, behavior descriptor={BD}")
    for _ in range(N_EXP):

        initial_genotype_size = NB_KEYPOINTS * GENE_PER_KEYPOINTS
        if CONTROLLER == 'interpolate keypoints end pause grip':
            initial_genotype_size = NB_KEYPOINTS * (GENE_PER_KEYPOINTS - 1) + 1
        if CONTROLLER == 'closed loop end pause grip':
            initial_genotype_size = GENES

        evaluation_function = bd_dict[BD]

        choose = None
        if ALGO == 'ns_rand_change_bd':
            choose = choose_evaluation_function

        if EVAL_SUCCESSFULL:
            QUALITY = False
            DISPLAY = True
            for j in range(2):
                for i in range(3):

                    path = os.path.join('runs', 'run59', 'type' + str(j) + '_' + str(i) + '.npy')
                    ind = np.load(path, allow_pickle=True)
                    res = eval_sucessfull_ind(ind)
                    before = res[0]
                    print('Individual did a succesfull graps ?', before)

                    if SAVE_TRAJ:
                        with open('traj.json', 'w') as outfile:
                            json.dump(res[1], outfile)

            exit()

        if EVAL_WITH_OBSTACLE:
            sphere_radius = 0.15
            obstacle_radius = 0.03
            random_obs = False
            if random_obs:
                n_obstacles = 20

                obstacle_pos = utils.sample_spherical(n_obstacles)
                obstacle_pos = obstacle_pos.transpose()
                obstacle_pos[obstacle_pos[:, 2] < 0] = -obstacle_pos[obstacle_pos[:, 2] < 0]
                obstacle_pos = sphere_radius * obstacle_pos
            
            else:
                n_obst_bins = 10
                obstacle_pos = utils.half_sphere_projection(r=sphere_radius, num=n_obst_bins)

            path_to_inds = glob.glob('')
            QUALITY = False

            inds = []
            for path in path_to_inds:
                ind = np.load(path, allow_pickle=True)
                inds.append(ind)
            
            count_sucess = np.zeros((len(obstacle_pos), len(inds)))
            for i, pos in enumerate(obstacle_pos):
                for j, ind in enumerate(inds):
                    res = eval_sucessfull_ind(ind, obstacle_pos=pos, obstacle_size=obstacle_radius)
                    if res[0]:
                        count_sucess[i, j] += 1
            np.save('results/obstacle_results.npy', count_sucess)
            exit()

        if EVAL_QUALITY:

            path_to_inds_qual = glob.glob('../qual_inds/*/*.npy')
            path_to_inds_no_qual = glob.glob('../no_qual_inds/*/*.npy')
            QUALITY = True

            inds_qual = []
            for path in path_to_inds_qual:
                ind = np.load(path, allow_pickle=True)
                inds_qual.append(ind)
            inds_no_qual = []
            for path in path_to_inds_no_qual:
                ind = np.load(path, allow_pickle=True)
                inds_no_qual.append(ind)
            
            quality_qual_inds = []
            quality_no_qual_inds = []
            for ind in inds_qual:
                res = evaluation_function(ind)
                qual = res[2]['grasp robustness']
                quality_qual_inds.append(qual)
            for ind in inds_no_qual:
                res = evaluation_function(ind)
                qual = res[2]['grasp robustness']
                quality_no_qual_inds.append(qual)

            qual = np.array(quality_qual_inds)
            no_qual = np.array(quality_no_qual_inds)
            print('Mean quality qual_inds:', np.mean(qual))
            print('Mean quality no_qual_inds:', np.mean(no_qual))
            np.save('results/qual.npy', qual)
            np.save('results/no_qual.npy', no_qual)

            exit()

        # deal with possible bootstrap
        boostrap_inds = None
        if BOOTSTRAP_FOLDER is not None:
            bootstrap_files = glob.glob(BOOTSTRAP_FOLDER + '*.npy')
            boostrap_inds = []
            for ind_file in bootstrap_files:
                ind = np.load(ind_file, allow_pickle=True)
                boostrap_inds.append(ind)
            print('Novelty Search boostrapped with ', len(boostrap_inds), ' individuals.')

        assert os.path.exists('runs'), "runs folder doesn't exist"
        res = None
        i = 0
        while res is None: # if res is None, it means the whole population is invalid (collision), so we do the search again
            t_start = time.time()
            res = noveltysearch.novelty_algo(evaluation_function, initial_genotype_size, BD_BOUNDS,
                                             mini=MINI,                           plot=PLOT,
                                             algo_type=ALGO,                      nb_gen=NB_GEN,
                                             bound_genotype=1,                    pop_size=POP_SIZE,
                                             parallelize=PARALLELIZE,             measures=True,
                                             choose_evaluate=choose,              bd_indexes=BD_INDEXES,
                                             archive_limit_size=ARCHIVE_LIMIT,    nb_cells=NB_CELLS,
                                             novelty_metric=NOVELTY_METRIC,       save_ind_cond='binary goal',
                                             bootstrap_individuals=boostrap_inds, multi_quality=MULTI_QUALITY_MEASURES,
                                             monitor_print=True)
            i += 1 # raise if failed 10 times
            if i>=10: raise Exception("The initial population failed 10 times")
        
        pop, archive, hof, details, figures, data, triumphant_archive = res
        print('Number of triumphants: ', len(triumphant_archive))
        
        i = 0 # create run directory
        while os.path.exists('runs/run%i/' % i):
            i += 1
        run_name = 'runs/run%i/' % i
        os.mkdir(run_name)
        
        # analyze triumphant archive diversity
        coverage, uniformity, clustered_triumphants = analyze_triumphants(triumphant_archive, run_name)
        t_end = time.time()

        # complete run dict
        details['run id'] = i
        details['controller'] = CONTROLLER
        details['object'] = OBJECT
        details['robot'] = ROBOT
        details['bootstrap folder'] = BOOTSTRAP_FOLDER
        details['run time'] = t_end - t_start
        if coverage is not None:
            details['successful'] = True
            details['diversity coverage'] = coverage
            details['diversity uniformity'] = uniformity
            details['number of successful'] = len(triumphant_archive)
        else:
            details['successful'] = False
        
        # direct plotting and saving figures
        if PLOT:
            fig = figures['figure']
            fig.savefig(run_name + 'novelty_search_plots.png')

            if ALGO == 'ns_rand_multi_bd':
                fig_4 = figures['figure_4']
                fig_4.savefig(run_name + 'eligibility_rates.png')

            if MULTI_QUALITY_MEASURES is not None:
                fig_3 = figures['figure_3']
                fig_3.savefig(run_name + 'qualities.png')
                
            if BD != 'change_bd':
                # plot final states
                archive_behavior = np.array([ind.behavior_descriptor.values for ind in archive]) if len(archive)>0 else None # archive can be empty if the offspring size is too small
                pop_behavior = np.array([ind.behavior_descriptor.values for ind in pop]) if len(pop)>0 else None
                hof_behavior = np.array([ind.behavior_descriptor.values for ind in hof]) if len(hof)>0 else None

            if archive_behavior is not None and pop_behavior is not None and hof_behavior is not None and len(archive_behavior[0]) == 2:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.set(title='Final position of object', xlabel='x', ylabel='y')
                ax.scatter(archive_behavior[:, 0], archive_behavior[:, 1], color='red', label='Archive')
                ax.scatter(pop_behavior[:, 0], pop_behavior[:, 1], color='blue', label='Population')
                ax.scatter(hof_behavior[:, 0], hof_behavior[:, 1], color='green', label='Hall of Fame')
                plt.legend()
                plt.savefig(run_name + 'bd_plot.png')
            elif archive_behavior is not None and pop_behavior is not None and hof_behavior is not None and len(archive_behavior[0]) == 3:
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(111, projection='3d')
                ax.set(title='Final position of object', xlabel='x', ylabel='y', zlabel='z')
                ax.scatter(archive_behavior[:, 0], archive_behavior[:, 1], archive_behavior[:, 2],
                           color='red', label='Archive')
                ax.scatter(pop_behavior[:, 0], pop_behavior[:, 1], pop_behavior[:, 2], color='blue', label='Population')
                ax.scatter(hof_behavior[:, 0], hof_behavior[:, 1], hof_behavior[:, 2],
                           color='green', label='Hall of Fame')
                plt.legend()
                plt.savefig(run_name + 'bd_plot.png')

            # plot genetic diversity
            gen_div_pop = np.array(data['population genetic statistics'])
            gen_div_off = np.array(data['offsprings genetic statistics'])
            if len(gen_div_pop[0]) <= 25:
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
                ax[0].set(title='Evolution of population genetic diversity', xlabel='Generations', ylabel='Std of gene')
                for i in range(len(gen_div_pop[0])):
                    if i < NB_KEYPOINTS * GENE_PER_KEYPOINTS:
                        color_index = i // GENE_PER_KEYPOINTS
                        rest = i % GENE_PER_KEYPOINTS
                        if rest == 0:
                            ax[0].plot(gen_div_pop[:, i], color=utils.color_list[color_index],
                                       label='keypoint ' + str(color_index))
                        else:
                            ax[0].plot(gen_div_pop[:, i], color=utils.color_list[color_index])
                    else:
                        color_index += 1
                        ax[0].plot(gen_div_pop[:, i], color=utils.color_list[color_index])
                ax[0].legend()
                ax[1].set(title='Evolution of offsprings genetic diversity', xlabel='Generations', ylabel='Std of gene')
                for i in range(len(gen_div_off[0])):
                    if i < NB_KEYPOINTS * GENE_PER_KEYPOINTS:
                        color_index = i // GENE_PER_KEYPOINTS
                        rest = i % GENE_PER_KEYPOINTS
                        if rest == 0:
                            ax[1].plot(gen_div_off[:, i], color=utils.color_list[color_index],
                                       label='keypoint ' + str(color_index))
                        else:
                            ax[1].plot(gen_div_off[:, i], color=utils.color_list[color_index])
                    else:
                        color_index += 1
                        ax[1].plot(gen_div_off[:, i], color=utils.color_list[color_index])
                ax[1].legend()
                plt.savefig(run_name + 'genetic_diversity_plot.png')
            #plt.show()
        # don't save some stuff
        if not SAVE_ALL:
            data['novelty distribution'] = None
            data['population genetic statistics'] = None
            data['offsprings genetic statistics'] = None

        # saving the run
        with open(run_name + 'run_details.json', 'w') as fp:
            json.dump(details, fp)
        with open(run_name + 'run_data.json', 'w') as fp:
            json.dump(data, fp)

        # display some individuals
        if DISPLAY_HOF:
            DISPLAY = True
            for ind in hof:
                evaluation_function(ind)
        
        if DISPLAY_RAND:
            DISPLAY = True
            nb_show = 10
            i = 0
            for ind in pop:
                if i <= nb_show:
                    evaluation_function(ind)
                else:
                    break
                i += 1

        if DISPLAY_TRIUMPHANTS:
            DISPLAY = True
            if clustered_triumphants is not None:
                for i in range(len(clustered_triumphants)):
                    print('Grasping type', i)
                    # show first 3 grasping of each types
                    for j in range(3):
                        if len(clustered_triumphants[i]) > j:
                            evaluation_function(clustered_triumphants[i][j])
    print("end")
