import gym
import noveltysearch
import utils
import numpy as np
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from deap import base, creator
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import controllers
import gym_grabbing
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
import operator

def greater(name, min, value):
    v = int(value)
    if v <= min: raise argparse.ArgumentTypeError(f"The {name.strip()} must be greater than {min}")
    return v

def cleanStr(x):
    return str(x).strip().lower()

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--robot", help="The robot environment", type=cleanStr, default="baxter", choices=["baxter", "kuka", "pepper", "crustcrawler", "kuka_iiwa_allegro", "ur10_shadow", "franka_panda"])
parser.add_argument("-o", "--object", help="The object to grasp", type=str, default="sphere")
parser.add_argument("-p", "--population", help="The poulation size", type=partial(greater, "population size", 1), default=96)
parser.add_argument("-g", "--generation", help="The number of generation", type=partial(greater, "number of generation", 1), default=1000)
parser.add_argument("-n", "--nruns", help="The number of time to repeat the search", type=partial(greater, "number of runs", 0), default=1)
parser.add_argument("-c", "--cells", help="The number of cells to measure the coverage", type=partial(greater, "number of cells", 1), default=1000)
parser.add_argument("-q", "--quality", help="Enable robustness as a quality: the individuals will be evaluated several times, thus the runs might be longer", action="store_true")
parser.add_argument("-k", "--keep-fail", help="Keep fails: it will log the run even if it fails", action="store_true")
parser.add_argument("-i", "--initial-random", help="Set reset_random_initial_object_pose to False, default to None", action="store_true")
parser.add_argument("-t", "--contact-table", help="Enable grasp success without touching the table", action="store_true")
parser.add_argument("-m", "--mode", help="Controller mode", type=str, default="joint positions", choices=["joint positions", "joint velocities", "joint torques", "inverse kinematics", "inverse dynamics", "pd stable"])
parser.add_argument("-b", "--bootstrap", help="Bootstrap folder", type=str, default=None)
parser.add_argument("-a", "--algorithm", help="Algorithm", type=cleanStr, default="qdmos", choices=["qdmos", "map-elites", "random", "ns", "ea"])
parser.add_argument("-e", "--early-stopping", help="Early stopping: the algorithm stops when the number of successes exceed the value", type=int, default=-1)
parser.add_argument("-d", "--behaviour-descriptor", help="The behaviour descriptor to use", type=cleanStr, default="pos_div_pos_grip", choices=["pos_div_grip", "pos_div_pos", "pos_div_pos_grip"])
parser.add_argument("-s", "--disable-state", help="Disable restore state: load each time the environment (slower) but it is more deterministic", action="store_false")
args = parser.parse_args()


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
RESET_MODE = args.disable_state

# choose parameters
POP_SIZE = args.population # -> 48 new individuals wil be evaluated each generation in order to match the nb of cores of MeSu beta with 2 nodes
NB_GEN = args.generation
OBJECT = args.object  # 'cuboid', 'mug.urdf', 'cylinder', 'deer.urdf', 'cylinder_r', 'glass.urdf'
ROBOT = args.robot  # 'baxter', 'pepper', 'kuka'
CONTROLLER = 'interpolate keypoints grip'#'dynamic movement primitives'#  # see controllers_dict for list
ALGO = {'qdmos':'ns_rand_multi_bd', 'ns':'ns_nov', 'map-elites':'map_elites', 'random':'random_search', 'ea':'classic_ea'}[args.algorithm] # algorithm
BD = args.behaviour_descriptor  # behavior descriptor type '2D', '3D', 'pos_div_grip', 'pos_div_pos_grip'
BOOTSTRAP_FOLDER = args.bootstrap
QUALITY = args.quality
AUTO_COLLIDE = True
NB_CELLS = args.cells; assert NB_CELLS>2  # number of cells for measurement
N_EXP = args.nruns



# controllers parameters
NB_KEYPOINTS = 3
PAUSE_FRAC = 0.66

ENV_KWARGS = {}
if ROBOT == 'baxter':
    ENV_NAME = 'baxter_grasping-v0'
    GENE_PER_KEYPOINTS = 7  # baxter is joints space: 8 joints
    LINK_ID_CONTACT = [47, 48, 49, 50, 51, 52]  # link ids that can have a grasping contact
    NB_STEPS_TO_ROLLOUT = 10
    NB_ITER = int(2000 / NB_STEPS_TO_ROLLOUT)
    ENV_KWARGS.update({'fixed_arm':RESET_MODE}) # best performance

elif ROBOT == 'pepper':
    ENV_NAME = 'pepper_grasping-v0'
    GENE_PER_KEYPOINTS = 6  # pepper is controlled in joints space: 7 joints
    LINK_ID_CONTACT = list(range(36, 50))  # link ids that can have a grasping contact
    NB_STEPS_TO_ROLLOUT = 1
    NB_ITER = int(1500 / NB_STEPS_TO_ROLLOUT)
    AUTO_COLLIDE = False


elif ROBOT in {'kuka', 'kuka_iiwa_allegro', 'franka_panda'}:
    ENV_NAME = {'kuka':'kuka_grasping-v0', 'kuka_iiwa_allegro':'kuka_iiwa_allegro-v0', 'franka_panda':'franka_panda-v0'}[args.robot]
    GENE_PER_KEYPOINTS = 7  # kuka is controlled in joints space: 7 joints
    LINK_ID_CONTACT = [8, 9, 10, 11, 12, 13]  # link ids that can have a grasping contact
    NB_STEPS_TO_ROLLOUT = 1
    NB_ITER = int((1500 if args.mode == "pd stable" else 2000) / NB_STEPS_TO_ROLLOUT)


elif ROBOT == 'crustcrawler':
    ENV_NAME = 'crustcrawler-v0'
    GENE_PER_KEYPOINTS = 6
    LINK_ID_CONTACT = [12,13,14]  # link ids that can have a grasping contact
    NB_STEPS_TO_ROLLOUT = 1
    NB_ITER = int(2500 / NB_STEPS_TO_ROLLOUT)

elif ROBOT == 'ur10_shadow':
    ENV_NAME = 'ur10_shadow-v0'
    GENE_PER_KEYPOINTS = 6
    NB_STEPS_TO_ROLLOUT = 1
    NB_ITER = int((1500 if args.mode == "pd stable" else 2000) / NB_STEPS_TO_ROLLOUT)
    AUTO_COLLIDE = False

# for closed_loop control
if ROBOT == 'baxter':
    GENES = 344
# TODO: implement closed loop control for pepper and kuka

# choose minor parameters
ADD_ITER = int(1*240/NB_STEPS_TO_ROLLOUT) # additional iteration: 1s
MINI = True  # minimization problem (used for MAP-elites)
DISTANCE_THRESH = 0.6  # is_success parameter
DIFF_OR_THRESH = 0.4  # threshold for clustering grasping orientations
COV_LIMIT = 0.1  # threshold for changing behavior descriptor in change_bd ns
N_LAG = int(200 / NB_STEPS_TO_ROLLOUT)  # number of steps before the grip time used in the pos_div_grip BD
ARCHIVE_LIMIT = 10000
N_REP_RAND = 20
DISPLACEMENT_RADIUS = 0.02 # radius of the displacement of the object during quality evaluation
ANGLE_NOISE = 10 / 180 * np.pi # the yaw rotation of the object will alternate with -ANGLE_NOISE and ANGLE_NOISE during robustness evaluation
FRICTION_NOISE = {'lateral':2, 'rolling':10, 'spinning':10} # the lateral friction of the object will be multiplied by FRICTION_NOISE and 1/FRICTION_NOISE when evaluating robustness
COUNT_SUCCESS = 0
NO_CONTACT_TABLE = args.contact_table

# precompute noise for robustness
rng = np.random.default_rng()
REPEAT_KWARGS = [None]*N_REP_RAND
for i in range(N_REP_RAND):
    d = float('inf')
    while d > 1:
        delta_pos = rng.random(size=(2,))*2-1
        d = np.linalg.norm(delta_pos)
    REPEAT_KWARGS[i] = dict(
        delta_pos=delta_pos*DISPLACEMENT_RADIUS,
        delta_yaw=ANGLE_NOISE*(2*rng.random()-1),
        multiply_friction={key:rng.choice([value, 1, 1/value]) for key, value in FRICTION_NOISE.items()},
    )

if QUALITY:
    D_POS = utils.circle_coordinates(N_REP_RAND, DISPLACEMENT_RADIUS)

if ALGO == 'ns_rand_aurora':
    N_SAMPLES = 4

ENV_KWARGS.update(dict(id=ENV_NAME, display=DISPLAY, obj=OBJECT, steps_to_roll=NB_STEPS_TO_ROLLOUT, mode=args.mode))
# if reset, create global env
# TODO: debug, for now RESET_MODE should be False
ENV = gym.make(**ENV_KWARGS, reset_random_initial_state=False if args.initial_random else None)


# choose diversity measure if gripping time is given by the controller
DIV_MEASURE = 'gripper orientation'  # 'gripper orientation', 'gripper orientation difference'

NOVELTY_METRIC = 'minkowski'

BD_INDEXES = None
MULTI_QUALITY_MEASURES = None # if it is left None, quality is not involved. By default, minimizing 'energy' is the quality
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
            MULTI_QUALITY_MEASURES = [['-energy'], ['+grasp robustness'], ['-energy']]
        else:
            MULTI_QUALITY_MEASURES = None#[['-energy'], ['-energy'], ['-energy']]
if BD == 'pos_div_pos':
    BD_BOUNDS = [[-0.35, 0.35], [-0.15, 0.2], [-0.2, 0.5], [-1, 1], [-1, 1], [-1, 1], [-1, 1],
                 [-0.35, 0.35], [-0.15, 0.2], [-0.2, 0.5]]
    if ALGO == 'ns_rand_multi_bd':
        BD_INDEXES = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        NOVELTY_METRIC = ['minkowski', 'minkowski', 'minkowski']
        if QUALITY:
            # MULTI_QUALITY_MEASURES = [['mean positive slope', 'grasp robustness', None], ['min', 'min', None]]
            MULTI_QUALITY_MEASURES = [['-energy'], ['+grasp robustness'], ['+grasp robustness']]
        else:
            MULTI_QUALITY_MEASURES = None#[['-energy'], ['-energy'], ['-energy']]
if BD == 'pos_div_pos_grip':
    BD_BOUNDS = [[-0.35, 0.35], [-0.15, 0.2], [-0.2, 0.5], [-1, 1], [-1, 1], [-1, 1], [-1, 1],
                 [-0.35, 0.35], [-0.15, 0.2], [-0.2, 0.5], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
    if ALGO == 'ns_rand_multi_bd':
        BD_INDEXES = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
        NOVELTY_METRIC = ['minkowski', 'minkowski', 'minkowski', 'minkowski']
        if QUALITY and NO_CONTACT_TABLE:
            MULTI_QUALITY_MEASURES = [['-energy'], ['+grasp robustness'], ['-energy'], ['-energy']]
        elif QUALITY:
            MULTI_QUALITY_MEASURES = [['-energy'], ['+grasp robustness'], ['+grasp robustness'], ['-energy']]
        else:
            MULTI_QUALITY_MEASURES = None #if args.mode=='joint torques' else [['-energy'], ['+reward'], ['-energy'], ['-energy']]
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


def diversity_measure(inf):
    if DIV_MEASURE == 'gripper orientation difference':
        # object orientation at gripping time (quaternion)
        obj_or = inf['object xyzw']

        # gripper orientation at gripping time (quaternion)
        grip_or = inf['end effector xyzw']

        # pybullet is x, y, z, w whereas pyquaternion is w, x, y, z
        obj_or = Quaternion(obj_or[3], obj_or[0], obj_or[1], obj_or[2])
        grip_or = Quaternion(grip_or[3], grip_or[0], grip_or[1], grip_or[2])

        # difference:
        measure = obj_or.conjugate * grip_or

    if DIV_MEASURE == 'gripper orientation':
        grip_or = inf['end effector xyzw']
        measure = Quaternion(grip_or[3], grip_or[0], grip_or[1], grip_or[2])

    return measure

def cluster_quaternion(triumphant_archive, max_size):
    n = len(triumphant_archive)
    if n < 2:
        return None, triumphant_archive
    rng = np.random.default_rng()
    if n > max_size and max_size>0: # subsample to reduce computation and memory
        archive = [triumphant_archive[i] for i in rng.choice(n, max_size, replace=False)]
        n = max_size
    else:
        archive = [triumphant_archive[i] for i in range(n)]

    q = np.array([m.info.values["diversity_descriptor"].unit.elements for m in archive])
    # cluster the triumphants with respect to grasping descriptor
    cluster = AgglomerativeClustering(
        n_clusters=None,
        affinity='precomputed',
        compute_full_tree=True,
        distance_threshold=DIFF_OR_THRESH,
        linkage='average'
    )

    # compute absolute_distance matrix in quaternion space, https://github.com/KieranWynn/pyquaternion/blob/99025c17bab1c55265d61add13375433b35251af/pyquaternion/quaternion.py#L772
    cluster = cluster.fit(np.min(np.linalg.norm([q+q[:,None], q-q[:,None]], axis=-1), axis=0))
    return cluster, archive

def callback(gen, archive, pop, max_size=10000, *args, **kwargs):
    if gen % 100 == 0:
        cluster, _ = cluster_quaternion(archive, max_size)
        return {'n clusters': cluster.n_clusters_ if cluster else len(archive)}
    else:
        return None

def analyze_triumphants(triumphant_archive, run_name, max_size=15000):
    if len(triumphant_archive) < 2:
        print('No individual completed the is_success.')
        return None, None, None, None

    # analyze the triumphants following the diversity descriptor
    measure = 'diversity_descriptor'

    # compute coverage and uniformity metrics: easy approach, use CVT cells in quaternion space
    bounds = [[-1, 1], [-1, 1], [-1, 1], [-1, 1]]
    # a quaternion is defined by a unit vector, so the quaternion space is a sphere
    cvt = utils.CVT(num_centroids=NB_CELLS, bounds=bounds, sphere=True)
    grid = np.zeros((NB_CELLS,))
    q = np.array([m.info.values[measure].unit.elements for m in triumphant_archive])
    indices, counts = np.unique(cvt.get_grid_index(q), return_counts=True)
    grid[indices] = counts
    coverage = np.count_nonzero(grid) / NB_CELLS
    uniformity = utils.compute_uniformity(grid).item()

    clustering, triumphant_archive = cluster_quaternion(triumphant_archive, max_size) # subsample

    number_of_clusters = clustering.n_clusters_
    labels = clustering.labels_

    indices, toSplit = np.nonzero(labels==np.arange(number_of_clusters)[:,None])
    indices = np.cumsum(np.count_nonzero(indices==np.arange(number_of_clusters-1)[:,None], axis=-1))
    clustered_triumphants = [[triumphant_archive[i] for i in subindices] for subindices in np.split(toSplit, indices)]

    print(number_of_clusters, 'types of grasping were found.')
    print('Coverage of', coverage, 'and uniformity of', uniformity)

    # saving the triumphants
    for i, clustered in enumerate(clustered_triumphants):
        # save first 3 grasping of each types
        if QUALITY:
            clustered = sorted(clustered, key=lambda ind: ind.info.values['grasp robustness'], reverse=True)
        else:
            clustered = sorted(clustered, key=lambda ind: ind.info.values['energy'], reverse=False)
        for j in range(3):
            if len(clustered) > j:
                ind = np.around(np.array(clustered[j]), 3)

                np.save(run_name + 'type' + str(i) + '_' + str(j), ind,
                        allow_pickle=True)
    xyzws = np.array([m.info.values['end effector xyzw relative object'] for m in triumphant_archive])
    xyzws /= np.linalg.norm(xyzws, axis=-1)[:,None]
    np.savez_compressed(
        file=run_name + 'individuals',
        genotypes=np.array(triumphant_archive),
        xyzws=xyzws, # quaternions
        positions=np.array([m.info.values['end effector position relative object'] for m in triumphant_archive]),
    ) # save all triumphants and infos

    return coverage, uniformity, clustered_triumphants, number_of_clusters


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
    else:
        ENV = gym.make(ENV_NAME, display=DISPLAY, obj=OBJECT, steps_to_roll=NB_STEPS_TO_ROLLOUT, object_position=OBJECT_POSITION, object_xyzw=OBJECT_XYZW)
    o = ENV.reset()

    individual = np.around(np.array(individual), 3)
    # initialize controller
    controller_info = controllers_info_dict[CONTROLLER]
    controller = controllers_dict[CONTROLLER](
        individual,
        **controller_info,
        initial=
            ENV.get_joint_state(position=True, normalized=True) if args.mode in {'joint positions', 'pd stable'} else
            ENV.get_joint_state(position=False, normalized=True) if args.mode == 'joint velocities' else None
    )


    for i in range(NB_ITER):
        # apply previously chosen action
        o, r, eo, info = ENV.step(controller.get_action(i, o))

        if eo:
            break
    # use last info to compute behavior and fitness
    behavior = [inf['object position'][0] - initial_object_position[0], inf['object position'][1] - initial_object_position[1]]  # last position of object

    # bound behavior descriptor on table
    utils.bound(behavior, BD_BOUNDS)

    # compute fitness
    fitness = utils.list_l2_norm(behavior, [0, 0])

    info = {}


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
    else:
        ENV = gym.make(ENV_NAME, display=DISPLAY, obj=OBJECT, steps_to_roll=NB_STEPS_TO_ROLLOUT, object_position=OBJECT_POSITION, object_xyzw=OBJECT_XYZW)

    o = ENV.reset()

    individual = np.around(np.array(individual), 3)

    # initialize controller
    controller_info = controllers_info_dict[CONTROLLER]
    controller = controllers_dict[CONTROLLER](
        individual,
        **controller_info,
        initial=
            ENV.get_joint_state(position=True, normalized=True) if args.mode in {'joint positions', 'pd stable'} else
            ENV.get_joint_state(position=False, normalized=True) if args.mode == 'joint velocities' else None
    )


    # for precise measure when we have the gripper assumption
    grabbed = False

    info = {}

    for i in range(NB_ITER):
        ENV.render()
        # apply previously chosen action
        o, r, eo, inf = ENV.step(controller.get_action(i, o))



        if i == 0:
            initial_object_position = inf['object position']

        if eo:
            break

        if hasattr(controller, 'grip_time'):
            # we are in the case where the gripping time is given
            # in consequence, we can do the precise measure of the grabbing orientation
            if action[-1] == -1 and not grabbed:
                # first action that orders the grabbing

                measure_grip_time = diversity_measure(inf)
                grabbed = True

    # use last info to compute behavior and fitness
    behavior = [inf['object position'][0] - initial_object_position[0], inf['object position'][1] - initial_object_position[1],
                inf['object position'][2]]  # last position of object

    # bound behavior descriptor on table
    if ALGO == 'ns_rand_change_bd':
        utils.bound(behavior, BD_BOUNDS[0])
    else:
        utils.bound(behavior, BD_BOUNDS)

    fitness = behavior[2] # compute fitness

    info['is_success'] = binary_goal = r # choose if individual satisfied the is_success

    if binary_goal:
        if hasattr(controller, 'grip_time'):
            info['diversity_descriptor'] = measure_grip_time
        else:
            # last object orientation (quaternion)
            obj_or = inf['object xyzw']

            # last gripper orientation (quaternion)
            grip_or = inf['end effector xyzw']

            # pybullet is x, y, z, w whereas pyquaternion is w, x, y, z
            obj_or = Quaternion(obj_or[3], obj_or[0], obj_or[1], obj_or[2])
            grip_or = Quaternion(grip_or[3], grip_or[0], grip_or[1], grip_or[2])

            # difference:
            diff_or = obj_or.conjugate * grip_or
            info['diversity_descriptor'] = diff_or


    return (behavior, (fitness,), info)


def pos_div_pos_grip_bd(individual):
    """Evaluates an individual: computes its value in the behavior descriptor space,
    and its fitness value.
    In this case, we consider the behavior space where we give the maximum amount of information
    as possible to the algorithm to establish a strong baseline.

    controller with grip_time is required

    4 descriptors:
    pos: the end position of the object in the 3D volume, always eligible
    div: the measure described by diversity_measure, eligible if the object is grabbed
    pos: the final position of the end effector, eligible if the object is grasped
    grip: the orientation of the gripper N_LAG steps before the gripping time, eligible if the object is touched

    This evaluation function can be used with either BD as:
    pos_div_pos
    pos_div_grip
    pos_div_pos_grip

    Args:
        individual (Individual): an individual

    Returns:
        tuple: tuple of behavior (list) fitness(tuple) info(dict)
    """
    global ENV
    o = ENV.reset(load='state' if RESET_MODE else 'all')

    global COUNT_SUCCESS

    individual = np.around(np.array(individual), 3)

    # initialize controller
    controller_info = controllers_info_dict[CONTROLLER]
    controller = controllers_dict[CONTROLLER](
        individual,
        **controller_info,
        initial=
            ENV.get_joint_state(position=True) if args.mode in {'joint positions', 'pd stable'} else
            ENV.get_joint_state(position=False) if args.mode == 'joint velocities' else None
    )
    assert(hasattr(controller, 'grip_time'))
    # lag_time = controller.grip_time - N_LAG
    lag_time = NB_ITER / 2



    # for precise measure when we have the gripper assumption
    already_touched = False
    already_grasped = False
    grasped_before_touch = False
    contact_robot_table = False
    closing = None

    # for measure at lag time
    lag_measured = False

    measure_grip_time = None
    pos_touch_time = None

    touch_idx = []

    # to compute quality for B1
    positive_dist_slope = 0
    prev_dist = None

    info, grip_info = {'energy':0, 'reward':0}, {"contact object table":[], "time close touch":float("inf")}

    if ALGO == 'map_elites':
        # define energy criterion
        energy = 0

    for i in range(NB_ITER):
        #ENV.render()
        # apply previously chosen action
        o, r, eo, inf = ENV.step(controller.get_action(i, o))

        if i == 0:
            initial_object_position = inf['object position']

        if eo: break

        if i >= controller.grip_time and not already_grasped:
            # first action that orders the gripper closure
            # measure_grip_time = diversity_measure(inf)
            already_grasped = True
            grip_info["contact object table"] = inf["contact object table"] # the object should touch the table while grasping
            closing = i # the object should be grasped right after closing the gripper


        if inf['touch'] and not already_touched:
            # first touch of object
            info['end effector position relative object'], info['end effector xyzw relative object'] = ENV.p.multiplyTransforms(
                *ENV.p.invertTransform(*ENV.p.getBasePositionAndOrientation(ENV.obj_id)),
                inf['end effector position'],
                inf['end effector xyzw']
            )
            measure_grip_time = diversity_measure(inf)
            pos_touch_time = inf['end effector position']
            already_touched = True
            if already_grasped:
                grasped_before_touch = True

        if inf['touch'] and closing: # get the time step difference between when start closing and touching
            grip_info["time close touch"] = i - closing
            closing = None

        if i >= lag_time and not lag_measured:
            # gripper orientation
            grip_or_lag = np.array(inf['end effector xyzw'])
            lag_measured = True

        if QUALITY: # quality 1 measured during the whole trajectory
            # only done one step after the start
            if prev_dist is None:
                # distance between gripper and object
                prev_dist = utils.list_l2_norm(inf['object position'], inf['end effector position'])
            else:
                new_dist = utils.list_l2_norm(inf['object position'], inf['end effector position'])
                differential_dist = new_dist - prev_dist
                if differential_dist > 0:
                    positive_dist_slope += differential_dist

                prev_dist = new_dist

        # if robot has a self-collision monitoring
        if AUTO_COLLIDE and inf['autocollision']:
            behavior = [None if ALGO=='ns_rand_multi_bd' else 0] * len(BD_BOUNDS)
            fitness = -float('inf')
            info = {'is_success': False, 'auto_collided': True}
            return (behavior, (fitness,), info)



        info['reward'] += r
        info['energy'] += np.abs(inf['applied joint motor torques']).sum()
        contact_robot_table = contact_robot_table or len(inf['contact robot table'])>0

    # use last info to compute behavior and fitness
    behavior = np.array([None]*len(BD_BOUNDS), dtype=object)
    behavior[:3] = [inf['object position'][0] - initial_object_position[0], inf['object position'][1] - initial_object_position[1], inf['object position'][2]]  # last position of object

    utils.bound(behavior[:3], BD_BOUNDS[:3])

    fitness = info['energy'] if ALGO == 'map_elites' else behavior[2] # compute fitness

    # choose if individual satisfied the is_success
    # the object should not touch the table neither the plane, must touch the gripper without penetration (with a margin of 0.005), be grasped right after closing the gripper (within 1s), touch the table when the gripper is closing
    grasp = r and grip_info['time close touch']<1*240/NB_STEPS_TO_ROLLOUT and len(grip_info['contact object table'])>0 # and not contact_robot_table

    if grasp: # there is maybe a grasp
        action_current_pos = np.hstack((ENV.get_joint_state(position=True, normalized=True), -1)) # get the current position + close the gripper
        if args.mode in {'joint positions', 'joint velocities', 'inverse kinematics'}: # set to position control
            ENV.env.mode = 'joint positions'
        else: # torque mode: motors are disabled so we use pd stable for position control in torque
            ENV.env.mode = 'pd stable'

        for i in range(ADD_ITER): # simulate
            o, r, eo, inf = ENV.step(action_current_pos) # the robot stops moving
        ENV.env.mode = args.mode # unset the mode
        grasp = r

    info['is_success'] = binary_goal = False
    if grasp:
        COUNT_SUCCESS += 1
        if measure_grip_time is None:
            # print('Individual grasped without touching any contact links')
            info['is_success'] = False
            pos_touch_time = None
        else:
            info['diversity_descriptor'] = measure_grip_time
            if (NO_CONTACT_TABLE and contact_robot_table) or (args.mode=='joint torques' and len(inf['contact robot table'])>0):
                measure_grip_time = None
            else:
                info['is_success'] = binary_goal = True
    else:
        measure_grip_time, pos_touch_time = None, None


    if already_touched: # this BD only active if trajectory touched the object
        grip_or_lag = np.array([grip_or_lag[3], *grip_or_lag[:3]]) # wxyz
    else:
        grip_or_lag = None

    behavior[3:7] = measure_grip_time # this one is common to the 3
    if BD == 'pos_div_pos':
        behavior[7:10] = pos_touch_time
    elif BD == 'pos_div_grip':
        behavior[7:11] = grip_or_lag
    elif BD == 'pos_div_pos_grip':
        behavior[7:10] = pos_touch_time
        behavior[10:] = grip_or_lag
    else:
        raise Exception(f"BD should be either: pos_div_pos, pos_div_grip or pos_div_pos_grip. BD={BD}")




    if ALGO != 'ns_rand_multi_bd':
        behavior = np.where(behavior==None, 0, behavior)
        return (behavior.tolist(), (fitness,), info)

    if QUALITY:
        if grasped_before_touch:
            # penalize more
            info['mean positive slope'] = 4 * positive_dist_slope / NB_ITER
        elif already_touched:
            info['mean positive slope'] = positive_dist_slope / NB_ITER
            if (not inf['closed gripper']) and inf['touch']:
                # gripper is not entirely closed at the end, and is touching the object
                info['mean positive slope'] -= 1
        else:
            info['mean positive slope'] = positive_dist_slope / NB_ITER + 1


    if QUALITY and binary_goal: # np.random.randint(2)
        info['repeat_kwargs'] = [{**repeat_kwargs, 'reference':np.array(inf['object position'])} for repeat_kwargs in REPEAT_KWARGS]
    return (behavior.tolist(), (fitness,), info)

def simulate(individual, delta_pos=[0,0], delta_yaw=0, multiply_friction={}, reference=None):

    global ENV
    o = ENV.reset(delta_pos=delta_pos, delta_yaw=delta_yaw, multiply_friction=multiply_friction, load='state' if RESET_MODE else 'all')

    # initialize controller
    controller_info = controllers_info_dict[CONTROLLER]
    controller = controllers_dict[CONTROLLER](
        individual,
        **controller_info,
        initial=
            ENV.get_joint_state(position=True, normalized=True) if args.mode in {'joint positions', 'pd stable'} else
            ENV.get_joint_state(position=False, normalized=True) if args.mode == 'joint velocities' else None
    )

    for i in range(NB_ITER):
        #ENV.render()
        o, r, eo, inf = ENV.step(controller.get_action(i, o))
        if eo: break

    if r: # there is maybe a grasp
        action_current_pos = np.hstack((ENV.get_joint_state(position=True, normalized=True), -1)) # get the current position + close the gripper
        if args.mode in {'joint positions', 'joint velocities', 'inverse kinematics'}: # set to position control
            ENV.env.mode = 'joint positions'
        else: # torque mode: motors are disabled so we use pd stable for position control in torque
            ENV.env.mode = 'pd stable'

        for i in range(ADD_ITER): # simulate
            o, r, eo, inf = ENV.step(action_current_pos) # the robot stops moving
        ENV.env.mode = args.mode # unset the mode



    if reference is not None:
        inf['distance to reference'] = np.linalg.norm(reference - inf['object position'])

    return inf

def reduce_repeat(ind, results):
    info = ind.info.values
    successes = 0
    mean_dist = 0
    for i, inf in enumerate(results):
        successes += inf['is_success']
        mean_dist += inf['distance to reference']
    info['grasp robustness'] = (successes + 1 / (1.00000001 + mean_dist/successes) if successes>0 else 0) / (i+1)
    #info.pop('repeat_kwargs')
    info['n is_success'] = successes

    return ind.behavior_descriptor.values, tuple(ind.fitness.values), info

def final_filter(ind, n=3): # filter n times because env.reset() is kind of stochastic
    for i in range(n):
        if not simulate(ind)['is_success']:
            return {'is_success': False}
    return {'is_success': True}

def eval_sucessfull_ind(individual, obstacle_pos=None, obstacle_size=None):

    if obstacle_pos is None:
        ENV = gym.make(ENV_NAME, display=DISPLAY, obj=OBJECT, steps_to_roll=NB_STEPS_TO_ROLLOUT, object_position=OBJECT_POSITION, object_xyzw=OBJECT_XYZW)
    else:
        ENV = gym.make(ENV_NAME, display=DISPLAY, obj=OBJECT, steps_to_roll=NB_STEPS_TO_ROLLOUT,
                       obstacle=True, obstacle_pos=obstacle_pos, obstacle_size=obstacle_size)
    o = ENV.reset()

    individual = np.around(np.array(individual), 3)

    # initialize controller
    controller_info = controllers_info_dict[CONTROLLER]
    controller = controllers_dict[CONTROLLER](
        individual,
        **controller_info,
        initial=
            ENV.get_joint_state(position=True, normalized=True) if args.mode in {'joint positions', 'pd stable'} else
            ENV.get_joint_state(position=False, normalized=True) if args.mode == 'joint velocities' else None
    )


    # for precise measure when we have the gripper assumption
    already_grasped = False

    count_keypoint = 0
    closed_keypoint_idx = 0
    traj_array = []
    closed = False

    for i in range(NB_ITER):
        ENV.render()
        # apply previously chosen action
        o, r, eo, inf = ENV.step(controller.get_action(i, o))


        if i % 15 == 0 and SAVE_TRAJ:
            count_keypoint += 1
            joint_config = o[7:7+GENE_PER_KEYPOINTS][10:17]
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
        if 'contact robot robot' in inf and AUTO_COLLIDE:
            if len(inf['contact robot robot']) != 0:
                raise Exception('Auto collision detected')

    binary_goal = r # choose if individual satisfied the is_success



    if SAVE_TRAJ:
        traj_array.append(closed_keypoint_idx)

    return binary_goal, traj_array


def aurora_bd(individual):

    if RESET_MODE:
        global ENV
    else:
        ENV = gym.make(ENV_NAME, display=DISPLAY, obj=OBJECT, steps_to_roll=NB_STEPS_TO_ROLLOUT, object_position=OBJECT_POSITION, object_xyzw=OBJECT_XYZW)

    o = ENV.reset()

    individual = np.around(np.array(individual), 3)

    # initialize controller
    controller_info = controllers_info_dict[CONTROLLER]
    controller = controllers_dict[CONTROLLER](
        individual,
        **controller_info,
        initial=
            ENV.get_joint_state(position=True, normalized=True) if args.mode in {'joint positions', 'pd stable'} else
            ENV.get_joint_state(position=False, normalized=True) if args.mode == 'joint velocities' else None
    )



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
        o, r, eo, inf = ENV.step(controller.get_action(i, o))

        if eo:
            break

        touch = len(inf['contact object robot']) > 0
        touch_id = 0
        if touch:
            touch_id = inf['contact object robot'][0][4]
            touch_idx.append(touch_id)
        relevant_touch = touch and (touch_id in LINK_ID_CONTACT)
        if relevant_touch and not already_touched:
            # first touch of object
            measure_grip_time = diversity_measure(inf)
            already_touched = True

        if i in sample_points:
            # we sample the trajectory to feed the high dimensional BD
            flattened_obs = [item for sublist in o for item in sublist]
            obs_bounds = [[-1, 1]] * 14 + [[-math.pi, math.pi]] * len(o[7:7+GENE_PER_KEYPOINTS])
            utils.bound(flattened_obs, obs_bounds)
            utils.normalize(flattened_obs, obs_bounds)
            behavior.append(flattened_obs)

    fitness = inf['object position'][2] # compute fitness

    info['is_success'] = binary_goal = r # choose if individual satisfied the is_success

    if binary_goal:
        if measure_grip_time is None:
            # print('Individual grasped without touching any contact links')
            info['is_success'] = False
        else:
            info['diversity_descriptor'] = measure_grip_time



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
                    'interpolate keypoints grip': controllers.InterpolateKeyPointsGrip,
                    'closed loop end pause grip': controllers.ClosedLoopEndPauseGripAssumption,
                    'dynamic movement primitives': controllers.DMPGripLift,
}
controllers_info_dict = {'interpolate keypoints end pause grip': {'pause_frac': PAUSE_FRAC, 'n_iter': NB_ITER,
                                                                  'nb_keypoints': NB_KEYPOINTS,
                                                                  'genes_per_keypoint': GENE_PER_KEYPOINTS},
                         'interpolate keypoints end pause': {'pause_frac': PAUSE_FRAC, 'n_iter': NB_ITER,
                                                             'NB_KEYPOINTS': NB_KEYPOINTS,
                                                             'GENE_PER_KEYPOINTS': GENE_PER_KEYPOINTS},
                         'discrete keypoints': {'n_iter': NB_ITER,
                                                'NB_KEYPOINTS': NB_KEYPOINTS,
                                                'GENE_PER_KEYPOINTS': GENE_PER_KEYPOINTS},
                         'interpolate keypoints': {'n_iter': NB_ITER,
                                                   'NB_KEYPOINTS': NB_KEYPOINTS,
                                                   'GENE_PER_KEYPOINTS': GENE_PER_KEYPOINTS},
                         'interpolate keypoints grip': {'n_iter': NB_ITER,
                                                   'nb_keypoints': NB_KEYPOINTS,
                                                   'genes_per_keypoint': GENE_PER_KEYPOINTS},
                         'closed loop end pause grip': {'n_iter': NB_ITER, 'pause_frac': PAUSE_FRAC},
                         'dynamic movement primitives': {'n_iter': NB_ITER, 'n_rollout': NB_STEPS_TO_ROLLOUT, 'τ':1}
}
bd_dict = {'2D': two_d_bd,
           '3D': three_d_bd,
           'pos_div_grip': pos_div_pos_grip_bd,
           'pos_div_pos_grip': pos_div_pos_grip_bd,
           'pos_div_pos': pos_div_pos_grip_bd,
           'aurora': aurora_bd}

if __name__ == "__main__":
    print(f"pop size={POP_SIZE}, ngen={NB_GEN}, object={OBJECT}, robot={ROBOT}, robustness={QUALITY}, autocollide={AUTO_COLLIDE}, nexp={N_EXP}, reset mode={RESET_MODE}, parallelize={PARALLELIZE}, controller={CONTROLLER}, behavior descriptor={BD}, mode={args.mode}, algorithm={ALGO}, random initial={args.initial_random}, keep fail={args.keep_fail}, early stopping={args.early_stopping}")
    if args.mode == 'pd stable' and os.environ.get('OPENBLAS_NUM_THREADS') != '1':
        print("WARNING: You better have to export OPENBLAS_NUM_THREADS to 1 in order to get the best performances when using 'pd stable' (np.linalg slows down with multiprocessing)")
    initial_state = ENV.get_state()

    for _ in range(N_EXP):
        if args.initial_random:
            ENV.new_random_initial_state()
            initial_state = ENV.get_state() # get the pose to share

        initial_genotype_size = NB_KEYPOINTS * GENE_PER_KEYPOINTS
        if CONTROLLER in {'interpolate keypoints end pause grip', 'interpolate keypoints grip'}:
            initial_genotype_size = NB_KEYPOINTS * GENE_PER_KEYPOINTS + 1
        elif CONTROLLER == 'closed loop end pause grip':
            initial_genotype_size = GENES
        elif CONTROLLER == "dynamic movement primitives":
            initial_genotype_size = 8+3*5 # 3(goal position)+4(goal orientation)+4(lift orientation)+3*5(xyz nb weights)

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
                            yaml.dump(res[1], outfile)

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
            bootstrap_files = Path(BOOTSTRAP_FOLDER).glob('type*.npy')
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
            res = noveltysearch.novelty_algo(
                evaluation_function, initial_genotype_size, BD_BOUNDS,
                mini=MINI,                           plot=PLOT,
                algo_type=ALGO,                      nb_gen=NB_GEN,
                bound_genotype=1,                    pop_size=POP_SIZE,
                parallelize=PARALLELIZE,             measures=True,
                choose_evaluate=choose,              bd_indexes=BD_INDEXES,
                archive_limit_size=ARCHIVE_LIMIT,    nb_cells=NB_CELLS,
                novelty_metric=NOVELTY_METRIC,       save_ind_cond='is_success',
                bootstrap_individuals=boostrap_inds, multi_quality=MULTI_QUALITY_MEASURES,
                monitor_print=True,                  final_filter=final_filter if RESET_MODE else None,
                repeat=simulate if QUALITY else None,reduce_repeat=reduce_repeat if QUALITY else None,
                early_stopping=args.early_stopping,  callback=callback,
            )
            i += 1 # raise if failed 10 times
            if i>=10: raise Exception("The initial population failed 10 times")

        pop, archive, hof, details, figures, data, triumphant_archive = res
        nb_of_triumphants = len(triumphant_archive)
        print('Number of triumphants: ', nb_of_triumphants)
        if len(triumphant_archive)==0 and not args.keep_fail: continue # do not report the logs

        i = 0 # create run directory
        while os.path.exists('runs/run%i/' % i):
            i += 1
        run_name = 'runs/run%i/' % i
        os.mkdir(run_name)

        # analyze triumphant archive diversity
        coverage, uniformity, clustered_triumphants, number_of_clusters = analyze_triumphants(triumphant_archive, run_name)
        t_end = time.time()

        # complete run dict
        details['robot'] = ROBOT
        details['run id'] = i
        details['controller'] = CONTROLLER
        details['bootstrap folder'] = BOOTSTRAP_FOLDER
        details['run time'] = t_end - t_start
        details['controller info'] = controllers_info_dict[CONTROLLER]
        details['behaviour descriptor'] = BD
        details['env kwargs'] = ENV_KWARGS
        details['initial state'] = initial_state
        if coverage is not None:
            details['successful'] = True
            details['diversity coverage'] = coverage
            details['diversity uniformity'] = uniformity
            details['number of successful'] = nb_of_triumphants
            details['number of clusters'] = number_of_clusters
            details['first success generation'] = data['first saved ind gen']
        else:
            details['successful'] = False
            details['number of successful'] = 0
            details['number of clusters'] = 0

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
        plt.close('all')
        # don't save some stuff
        if not SAVE_ALL:
            data['novelty distribution'] = None
            data['population genetic statistics'] = None
            data['offsprings genetic statistics'] = None


        # saving the run
        utils.save_yaml(details, run_name + 'run_details.yaml')

        # cleaning data
        data = {key:value for key, value in data.items() if not value is None or isinstance(value, np.ndarray) and value.dtype == np.dtype(object)}
        np.savez_compressed(run_name+'run_data', **data) # or maybe save to parquet as a dataFrame

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

    if RESET_MODE:
        ENV.close()
    print("end")
