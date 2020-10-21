import gym
import noveltysearch
import utils
import numpy as np
import random
import pyquaternion as pyq
import matplotlib.pyplot as plt
from deap import base, creator
from sklearn.cluster import AgglomerativeClustering
import controllers

DISPLAY = False
PARALLELIZE = True
PLOT = True
DISPLAY_HOF = False
DISPLAY_RAND = False
DISPLAY_TRIUMPHANTS = True
CLASSIFY = False

# choose parameters
NB_KEYPOINTS = 3
NB_ITER = 5000
MINI = False  # maximization problem
HEIGHT_THRESH = -0.08  # binary goal parameter
DISTANCE_THRESH = 0.20  # binary goal parameter
DIFF_OR_THRESH = 0.1  # threshold for clustering grasping orientations

# choose behavior descriptor type
BD = '3D'

if BD == '2D':
    BD_BOUNDS = [[-0.35, 0.35], [-0.15, 0.2]]
if BD == '3D':
    BD_BOUNDS = [[-0.35, 0.35], [-0.15, 0.2], [-0.12, 0.5]]

# choose controller type
CONTROLLER = 'interpolate keypoints'

# choose algorithm type
ALGO = 'ns_rand'
    

if PARALLELIZE:
    # container for behavior descriptor
    creator.create('BehaviorDescriptor', list)
    # container for info
    creator.create('Info', dict)
    # container for novelty
    creator.create('Novelty', base.Fitness, weights=(1.0,))
    # container for fitness
    if MINI:
        creator.create('Fit', base.Fitness, weights=(-1.0,))
    else:
        creator.create('Fit', base.Fitness, weights=(1.0,))

    # container for individual
    creator.create('Individual', list, behavior_descriptor=creator.BehaviorDescriptor,
                   novelty=creator.Novelty, fitness=creator.Fit, info=creator.Info)

    # set creator
    noveltysearch.set_creator(creator)


def analyze_triumphants(triumphant_archive):
    if len(triumphant_archive) < 2:
        print('No individual completed the binary goal.')
        return None

    # sample the triumphant archive to reduce computational cost
    while len(triumphant_archive) >= 500:
        triumphant_archive.pop(random.randint(0, len(triumphant_archive)))
    
    nb_of_triumphants = len(triumphant_archive)

    # cluster the triumphants with respect to grasping descriptor
    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', compute_full_tree=True,
                                         distance_threshold=0.1)
    # compute distance matrix
    X = np.zeros((nb_of_triumphants, nb_of_triumphants))
    for x in range(nb_of_triumphants):
        for y in range(nb_of_triumphants):
            if x == y:
                X[x, y] = 0
            else:
                triumphant_a = triumphant_archive[x].info.values['orientation difference']
                triumphant_b = triumphant_archive[y].info.values['orientation difference']
                X[x, y] = pyq.Quaternion.absolute_distance(triumphant_a, triumphant_b)

    # fit distance matrix
    clustering.fit(X)

    number_of_clusters = clustering.n_clusters_
    labels = clustering.labels_

    clustered_triumphants = []
    for i in range(number_of_clusters):
        clustered_triumphants.append(triumphant_archive[labels[i]])

    print(number_of_clusters, 'types of grasping were found.')

    return clustered_triumphants
    

def two_d_behavioral_descriptor(individual):
    """Evaluates an individual: computes its value in the behavior descriptor space,
    and its fitness value.
    In this case, we consider the behavior space as the end position of the object in
    the 2D plane (no grasping needed, only pushing).
    The genotype represents NB_KEYPOINTS keypoints that are a triplet representing the position
    of the gripper.
    Then, each keypoint is given as goal sequentially for the same amount of time.

    Args:
        individual (Individual): an individual

    Returns:
        tuple: tuple of behavior (list) and fitness(tuple)
    """
    env = gym.make('gym_baxter_grabbing:baxter_grabbing-v0', display=DISPLAY)
    actions = []
    for i in range(NB_KEYPOINTS):
        actions.append(individual[4 * i:4 * (i + 1)])

    action = actions[0]

    # initialize controller
    controller = controllers_dict[CONTROLLER](actions, NB_ITER)

    for i in range(NB_ITER):
        env.render()
        # apply previously chosen action
        o, r, eo, info = env.step(action)

        if i == 0:
            initial_object_position = o[0]

        action = controller.get_action(i)

        if eo:
            break
    # use last info to compute behavior and fitness
    behavior = [o[0][0] - initial_object_position[0], o[0][1] - initial_object_position[1]]  # last position of object

    # bound behavior descriptor on table
    utils.bound(behavior, BD_BOUNDS)

    # compute fitness
    fitness = utils.list_l2_norm(behavior, [0, 0])

    info = {}

    env.close()
    return (behavior, (fitness,), info)


def three_d_behavioral_descriptor(individual):
    """Evaluates an individual: computes its value in the behavior descriptor space,
    and its fitness value.
    In this case, we consider the behavior space as the end position of the object in
    the 3D volume.
    The genotype represents NB_KEYPOINTS keypoints that are a triplet representing the position
    of the gripper.
    Then, each keypoint is given as goal sequentially for the same amount of time.

    Args:
        individual (Individual): an individual

    Returns:
        tuple: tuple of behavior (list) and fitness(tuple)
    """
    env = gym.make('gym_baxter_grabbing:baxter_grabbing-v0', display=DISPLAY)
    actions = []
    for i in range(NB_KEYPOINTS):
        actions.append(individual[4 * i:4 * (i + 1)])

    action = actions[0]

    # initialize controller
    controller = controllers_dict[CONTROLLER](actions, NB_ITER)

    for i in range(NB_ITER):
        env.render()
        # apply previously chosen action
        o, r, eo, info = env.step(action)

        if i == 0:
            initial_object_position = o[0]

        action = controller.get_action(i)

        if eo:
            break
    # use last info to compute behavior and fitness
    behavior = [o[0][0] - initial_object_position[0], o[0][1] - initial_object_position[1],
                o[0][2]]  # last position of object

    # bound behavior descriptor on table
    utils.bound(behavior, BD_BOUNDS)

    # compute fitness
    fitness = behavior[2]

    info = {}

    # choose if individual satisfied the binary goal
    dist = utils.list_l2_norm(o[0], o[2])
    binary_goal = False
    if o[0][2] > HEIGHT_THRESH and dist < DISTANCE_THRESH:
        binary_goal = True
    info['binary goal'] = binary_goal

    if binary_goal:
        # last object orientation (quaternion)
        obj_or = o[1]

        # last gripper orientation (quaternion)
        grip_or = o[3]

        # pybullet is x, y, z, w whereas pyquaternion is w, x, y, z
        obj_or = pyq.Quaternion(obj_or[3], obj_or[0], obj_or[1], obj_or[2])
        grip_or = pyq.Quaternion(grip_or[3], grip_or[0], grip_or[1], grip_or[2])

        # difference:
        diff_or = obj_or.conjugate * grip_or
        info['orientation difference'] = diff_or

    env.close()
    return (behavior, (fitness,), info)


controllers_dict = {'discrete keypoints': controllers.ControllerDiscreteKeyPoints,
                    'interpolate keypoints': controllers.ControllerInterpolateKeyPoints}

bd_dict = {'2D': two_d_behavioral_descriptor,
           '3D': three_d_behavioral_descriptor}

if __name__ == "__main__":

    initial_genotype_size = NB_KEYPOINTS * 4
    evaluation_function = bd_dict[BD]
    pop, archive, hof, info = noveltysearch.novelty_algo(evaluation_function, initial_genotype_size, BD_BOUNDS,
                                                         mini=MINI,
                                                         plot=PLOT, algo_type=ALGO, nb_gen=100, bound_genotype=1,
                                                         pop_size=100, parallelize=PARALLELIZE, measures=True)
    
    # create triumphant archive
    triumphant_archive = []
    for ind in archive:
        if ind.info.values['binary goal']:
            triumphant_archive.append(ind)
    
    # analyze triumphant archive diversity
    clustered_triumphants = analyze_triumphants(triumphant_archive)

    if PLOT:
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
        if BD == '3D':
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.set(title='Final position of object', xlabel='x', ylabel='y', zlabel='z')
            ax.scatter(archive_behavior[:, 0], archive_behavior[:, 1], archive_behavior[:, 2],
                       color='red', label='Archive')
            ax.scatter(pop_behavior[:, 0], pop_behavior[:, 1], pop_behavior[:, 2], color='blue', label='Population')
            ax.scatter(hof_behavior[:, 0], hof_behavior[:, 1], hof_behavior[:, 2], color='green', label='Hall of Fame')
        plt.legend()

    plt.show()
    
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
                        np.save('grasping_type' + str(i) + '_' + str(j), np.array(clustered_triumphants[i][j]),
                                allow_pickle=True)
