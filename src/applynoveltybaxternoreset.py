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

DISPLAY = False
PARALLELIZE = True
PLOT = True
DISPLAY_HOF = False
DISPLAY_RAND = False
DISPLAY_TRIUMPHANTS = False
EVAL_SUCCESSFULL = False
SAVE_ALL = False
RESET_MODE = True


# choose parameters
POP_SIZE = 100
NB_GEN = 100
NB_KEYPOINTS = 3
GENE_PER_KEYPOINTS = 7
GENES = 343
ADDITIONAL_GENES = 1  # minimum  1 for the gripper assumption
PAUSE_FRAC = 0.66
NB_STEPS_TO_ROLLOUT = 10
NB_ITER = int(6000 / NB_STEPS_TO_ROLLOUT)  # 6000 divided by 10, where 10 is the nb_of_steps_to_roll in the gym env
MINI = False  # maximization problem
HEIGHT_THRESH = -0.08  # binary goal parameter
DISTANCE_THRESH = 0.20  # binary goal parameter
DIFF_OR_THRESH = 0.4  # threshold for clustering grasping orientations
COV_LIMIT = 0.1  # threshold for changing behavior descriptor in change_bd ns
N_LAG = int(200 / NB_STEPS_TO_ROLLOUT)  # number of steps before the grip time used in the multi_full_info BD
ARCHIVE_LIMIT = 2500
NB_CELLS = 100  # number of cells for measurement

# if reset, create global env
if RESET_MODE:
    ENV = gym.make('gym_baxter_grabbing:baxter_grabbing-v1', display=DISPLAY)
    ENV.set_steps_to_roll(NB_STEPS_TO_ROLLOUT)

# choose controller type
CONTROLLER = 'interpolate keypoints end pause grip'

# choose diversity measure if gripping time is given by the controller
DIV_MEASURE = 'gripper orientation'  # 'gripper orientation', 'gripper orientation difference'

# choose algorithm type
ALGO = 'ns_rand_multi_bd'

# choose behavior descriptor type
BD = 'multi_full_info'

NOVELTY_METRIC = 'minkowski'

BD_INDEXES = None
if BD == '2D':
    BD_BOUNDS = [[-0.35, 0.35], [-0.15, 0.2]]
if BD == '3D':
    BD_BOUNDS = [[-0.35, 0.35], [-0.15, 0.2], [-0.12, 0.5]]
if BD == 'multi':
    BD_BOUNDS = [[-0.35, 0.35], [-0.15, 0.2], [-0.12, 0.5], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
    BD_INDEXES = [0, 0, 0, 1, 1, 1, 1]
    if ALGO == 'ns_rand_multi_bd':
        NOVELTY_METRIC = ['minkowski', utils.quatmetric]
if BD == 'multi_full_info':
    BD_BOUNDS = [[-0.35, 0.35], [-0.15, 0.2], [-0.12, 0.5], [-1, 1], [-1, 1], [-1, 1], [-1, 1],
                 [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
    BD_INDEXES = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    if ALGO == 'ns_rand_multi_bd':
        NOVELTY_METRIC = ['minkowski', utils.quatmetric, utils.quatmetric]
if ALGO == 'ns_rand_change_bd':
    BD = 'change_bd'
    # list of 3D bd and orientation bd
    BD_BOUNDS = [[[-0.35, 0.35], [-0.15, 0.2], [-0.12, 0.5]], [[-1, 1], [-1, 1], [-1, 1], [-1, 1]]]

if PARALLELIZE:
    # container for behavior descriptor
    creator.create('BehaviorDescriptor', list)
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
                   gen_info=creator.GenInfo)

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

    # # sample the triumphant archive to reduce computational cost
    # no need since the archive size is now bounded
    # while len(triumphant_archive) >= 1000:
    #     triumphant_archive.pop(random.randint(0, len(triumphant_archive) - 1))
    
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

                # debug
                evaluation_function = bd_dict[BD]
                res = evaluation_function(ind)
                assert(res[2]['binary goal'])

                np.save(run_name + 'type' + str(i) + '_' + str(j), ind,
                        allow_pickle=True)

                # debug
                ind_2 = np.load(run_name + 'type' + str(i) + '_' + str(j) + '.npy', allow_pickle=True)
                res_2 = evaluation_function(ind_2)
                assert(res_2[2]['binary goal'])

    return coverage, uniformity, clustered_triumphants
    

def two_d_behavioral_descriptor(individual):
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
    ENV = gym.make('gym_baxter_grabbing:baxter_grabbing-v1', display=DISPLAY)
    ENV.set_steps_to_roll(NB_STEPS_TO_ROLLOUT)
    individual = np.around(np.array(individual), 3)
    # initialize controller
    controller_info = controllers_info_dict[CONTROLLER]
    controller = controllers_dict[CONTROLLER](individual, controller_info)
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


def three_d_behavioral_descriptor(individual):
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
    ENV = gym.make('gym_baxter_grabbing:baxter_grabbing-v1', display=DISPLAY)
    ENV.set_steps_to_roll(NB_STEPS_TO_ROLLOUT)
    individual = np.around(np.array(individual), 3)
    
    # initialize controller
    controller_info = controllers_info_dict[CONTROLLER]
    controller = controllers_dict[CONTROLLER](individual, controller_info)
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


def multi_full_behavior_descriptor(individual):
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
        ENV = gym.make('gym_baxter_grabbing:baxter_grabbing-v1', display=DISPLAY)
        ENV.set_steps_to_roll(NB_STEPS_TO_ROLLOUT)
        individual = np.around(np.array(individual), 3)

    # initialize controller
    controller_info = controllers_info_dict[CONTROLLER]
    controller = controllers_dict[CONTROLLER](individual, controller_info)
    assert(hasattr(controller, 'grip_time'))
    lag_time = controller.grip_time - N_LAG
    action = controller.initial_action

    # for precise measure when we have the gripper assumption
    grabbed = False

    # for measure at lag time
    lag_measured = False

    measure_grip_time = None

    touch_idx = []

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
        
        # version 1: measure is done at grip_time
        # if action[-1] == -1 and not grabbed:
        #     # first action that orders the grabbing
        #     measure_grip_time = diversity_measure(o)
        #     grabbed = True

        # version 2: measure is done at touch time

        touch = len(inf['contact_points']) > 0
        touch_id = 0
        if touch:
            touch_id = inf['contact_points'][0][3]
            touch_idx.append(touch_id)
        relevant_touch = touch and touch_id >= 49
        if relevant_touch and not grabbed:
            # first touch of object
            measure_grip_time = diversity_measure(o)
            grabbed = True
        
        if i >= lag_time and not lag_measured:
            # gripper orientation
            grip_or_lag = o[3]
            lag_measured = True

    # use last info to compute behavior and fitness
    behavior = [o[0][0] - initial_object_position[0], o[0][1] - initial_object_position[1],
                o[0][2]]  # last position of object

    utils.bound(behavior, BD_BOUNDS[0:3])
    # set bd values between -1 and 1
    utils.normalize(behavior, BD_BOUNDS[0:3])

    # append 4 times None to behavior in case no grabbing (modified later)
    for _ in range(4):
        behavior.append(None)

    # compute fitness
    fitness = behavior[2]

    # choose if individual satisfied the binary goal
    dist = utils.list_l2_norm(o[0], o[2])
    binary_goal = False
    if o[0][2] > HEIGHT_THRESH and dist < DISTANCE_THRESH:
        binary_goal = True
    info['binary goal'] = binary_goal

    if binary_goal:
        assert(measure_grip_time is not None)
        info['diversity_descriptor'] = measure_grip_time
        behavior[3] = measure_grip_time[0]  # Quat to array
        behavior[4] = measure_grip_time[1]
        behavior[5] = measure_grip_time[2]
        behavior[6] = measure_grip_time[3]
    
    behavior.append(grip_or_lag[3])
    behavior.append(grip_or_lag[0])
    behavior.append(grip_or_lag[1])
    behavior.append(grip_or_lag[2])

    if not RESET_MODE:
        ENV.close()
    
    return (behavior, (fitness,), info)


def multi_behavioral_descriptor(individual):
    """Evaluates an individual: computes its value in the behavior descriptor space,
    and its fitness value.
    In this case, we consider two behavioral descriptors:
    -the end position of the object in the 3D volume, always eligible
    -the measure described by diversity_measure, eligible if the object is grabbed

    Args:
        individual (Individual): an individual

    Returns:
        tuple: tuple of behavior (list) and fitness(tuple)
    """
    if RESET_MODE:
        global ENV
        ENV.reset()
    ENV = gym.make('gym_baxter_grabbing:baxter_grabbing-v1', display=DISPLAY)
    ENV.set_steps_to_roll(NB_STEPS_TO_ROLLOUT)
    individual = np.around(np.array(individual), 3)

    # initialize controller
    controller_info = controllers_info_dict[CONTROLLER]
    controller = controllers_dict[CONTROLLER](individual, controller_info)
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

    utils.bound(behavior, BD_BOUNDS[0:3])
    # set bd values between -1 and 1
    utils.normalize(behavior, BD_BOUNDS[0:3])

    # append 4 times None to behavior in case no grabbing (modified later)
    for _ in range(4):
        behavior.append(None)

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
            behavior[3] = measure_grip_time[0]  # Quat to array
            behavior[4] = measure_grip_time[1]
            behavior[5] = measure_grip_time[2]
            behavior[6] = measure_grip_time[3]
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
            behavior[3] = diff_or[0]  # Quat to array
            behavior[4] = diff_or[1]
            behavior[5] = diff_or[2]
            behavior[6] = diff_or[3]

    if not RESET_MODE:
        ENV.close()
    return (behavior, (fitness,), info)


def orientation_behavioral_descriptor(individual):
    """Evaluates an individual: computes its value in the behavior descriptor space,
    and its fitness value.
    In this case, we consider the behavior space as the gripping orientation.

    Args:
        individual (Individual): an individual

    Returns:
        tuple: tuple of behavior (list) and fitness(tuple)
    """
    if RESET_MODE:
        global ENV
        ENV.reset()
    ENV = gym.make('gym_baxter_grabbing:baxter_grabbing-v1', display=DISPLAY)
    ENV.set_steps_to_roll(NB_STEPS_TO_ROLLOUT)
    individual = np.around(np.array(individual), 3)

    # initialize controller
    controller_info = controllers_info_dict[CONTROLLER]
    controller = controllers_dict[CONTROLLER](individual, controller_info)
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
    
    # bound behavior descriptor on table because we want to keep the same computation of fitness
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

    # re attribute the behavior
    if binary_goal:
        if hasattr(controller, 'grip_time'):
            info['diversity_descriptor'] = measure_grip_time
            # Quat to array
            behavior = [measure_grip_time[0], measure_grip_time[1], measure_grip_time[2], measure_grip_time[3]]
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
            behavior = [diff_or[0], diff_or[1], diff_or[2], diff_or[3]]  # Quat to array

    else:
        # set behavior as None and deal with in novelty search
        behavior = None

    if not RESET_MODE:
        ENV.close()
    return (behavior, (fitness,), info)


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
                                                                  'GENE_PER_KEYPOINTS': GENE_PER_KEYPOINTS,
                                                                  'ADDITIONAL_GENES': ADDITIONAL_GENES},
                         'interpolate keypoints end pause': {'pause_frac': PAUSE_FRAC, 'n_iter': NB_ITER,
                                                             'NB_KEYPOINTS': NB_KEYPOINTS,
                                                             'GENE_PER_KEYPOINTS': GENE_PER_KEYPOINTS},
                         'discrete keypoints': {'n_iter': NB_ITER,
                                                'NB_KEYPOINTS': NB_KEYPOINTS,
                                                'GENE_PER_KEYPOINTS': GENE_PER_KEYPOINTS},
                         'interpolate keypoints': {'n_iter': NB_ITER,
                                                   'NB_KEYPOINTS': NB_KEYPOINTS,
                                                   'GENE_PER_KEYPOINTS': GENE_PER_KEYPOINTS},
                         'closed loop end pause grip': {'n_iter': NB_ITER, 'pause_frac': PAUSE_FRAC,
                                                        'ADDITIONAL_GENES': ADDITIONAL_GENES}}

bd_dict = {'2D': two_d_behavioral_descriptor,
           '3D': three_d_behavioral_descriptor,
           'change_bd': [three_d_behavioral_descriptor, orientation_behavioral_descriptor],
           'multi': multi_behavioral_descriptor,
           'multi_full_info': multi_full_behavior_descriptor}

if __name__ == "__main__":

    initial_genotype_size = NB_KEYPOINTS * GENE_PER_KEYPOINTS + ADDITIONAL_GENES
    if CONTROLLER == 'closed loop end pause grip':
        initial_genotype_size = GENES + ADDITIONAL_GENES
    evaluation_function = bd_dict[BD]

    choose = None
    if ALGO == 'ns_rand_change_bd':
        choose = choose_evaluation_function

    if EVAL_SUCCESSFULL:

        for j in range(4):
            for i in range(3):
                path = os.path.join('../exp_results/106', 'run11', 'type' + str(j) + '_' + str(i) + '.npy')
                ind = np.load(path, allow_pickle=True)
                res = evaluation_function(ind)
                before = res[2]['binary goal']
                print(before)
                res_2 = evaluation_function(ind)
                after = res_2[2]['binary goal']
                assert(before == after)

        exit()
    
    i = 0
    while os.path.exists('runs/run%i/' % i):
        i += 1
    run_name = 'runs/run%i/' % i
    os.mkdir(run_name)

    res = noveltysearch.novelty_algo(evaluation_function, initial_genotype_size, BD_BOUNDS,
                                     mini=MINI, plot=PLOT, algo_type=ALGO,
                                     nb_gen=NB_GEN, bound_genotype=1,
                                     pop_size=POP_SIZE, parallelize=PARALLELIZE,
                                     measures=True,
                                     choose_evaluate=choose, bd_indexes=BD_INDEXES,
                                     archive_limit_size=ARCHIVE_LIMIT, nb_cells=NB_CELLS,
                                     novelty_metric=NOVELTY_METRIC)
    
    pop, archive, hall_of_fame, details, figures, data = res
    
    # create triumphant archive
    triumphant_archive = []
    for ind in archive:
        if ind.info.values['binary goal']:
            triumphant_archive.append(ind)
    
    # analyze triumphant archive diversity
    coverage, uniformity, clustered_triumphants = analyze_triumphants(triumphant_archive, run_name)

    # complete run dict
    details['run id'] = i
    details['controller'] = CONTROLLER
    if coverage is not None:
        details['successful'] = True
        details['diversity coverage'] = coverage
        details['diversity uniformity'] = uniformity
    else:
        details['successful'] = False
    
    # don't save some stuff
    if not SAVE_ALL:
        data['novelty distribution'] = None
        data['population genetic statistics'] = None
        data['offsprings genetic statistics'] = None

    # direct plotting and saving figures
    if PLOT:
        fig = figures['figure']
        fig.savefig(run_name + 'novelty_search_plots.png')
        if BD != 'change_bd':
            # plot final states
            archive_behavior = np.array([ind.behavior_descriptor.values for ind in archive])
            pop_behavior = np.array([ind.behavior_descriptor.values for ind in pop])
            hof_behavior = np.array([ind.behavior_descriptor.values for ind in hof])

        if BD == '2D':
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set(title='Final position of object', xlabel='x', ylabel='y')
            ax.scatter(archive_behavior[:, 0], archive_behavior[:, 1], color='red', label='Archive')
            ax.scatter(pop_behavior[:, 0], pop_behavior[:, 1], color='blue', label='Population')
            ax.scatter(hof_behavior[:, 0], hof_behavior[:, 1], color='green', label='Hall of Fame')
            plt.legend()
            plt.savefig(run_name + 'bd_plot.png')
        if BD == '3D':
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.set(title='Final position of object', xlabel='x', ylabel='y', zlabel='z')
            ax.scatter(archive_behavior[:, 0], archive_behavior[:, 1], archive_behavior[:, 2],
                       color='red', label='Archive')
            ax.scatter(pop_behavior[:, 0], pop_behavior[:, 1], pop_behavior[:, 2], color='blue', label='Population')
            ax.scatter(hof_behavior[:, 0], hof_behavior[:, 1], hof_behavior[:, 2], color='green', label='Hall of Fame')
            plt.legend()
            plt.savefig(run_name + 'bd_plot.png')

        # plot genetic diversity
        gen_div_pop = np.array(run['population genetic statistics'])
        gen_div_off = np.array(run['offsprings genetic statistics'])
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
        plt.show()
    
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
