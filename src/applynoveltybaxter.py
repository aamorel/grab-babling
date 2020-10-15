import gym
import noveltysearch
import utils
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator
import controllers

DISPLAY = False
PARALLELIZE = True
PLOT = True
DISPLAY_HOF = False
DISPLAY_RAND = True

# choose parameters
NB_KEYPOINTS = 3
NB_ITER = 5000
MINI = False  # maximization problem

# choose behavior descriptor type
BD = '2D'

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
    # container for novelty
    creator.create('Novelty', base.Fitness, weights=(1.0,))
    # container for fitness
    if MINI:
        creator.create('Fit', base.Fitness, weights=(-1.0,))
    else:
        creator.create('Fit', base.Fitness, weights=(1.0,))

    # container for individual
    creator.create('Individual', list, behavior_descriptor=creator.BehaviorDescriptor,
                   novelty=creator.Novelty, fitness=creator.Fit)

    # set creator
    noveltysearch.set_creator(creator)


def list_l2_norm(list1, list2):
    if len(list1) != len(list2):
        raise NameError('The two lists have different length')
    else:
        dist = 0
        for i in range(len(list1)):
            dist += (list1[i] - list2[i]) ** 2
        dist = dist ** (1 / 2)
        return dist


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
    fitness = list_l2_norm(behavior, [0, 0])

    env.close()
    return (behavior, (fitness,))


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
    behavior = [o[0][0] - initial_object_position[0], o[0][1] - initial_object_position[1]]  # last position of object

    # bound behavior descriptor on table
    utils.bound(behavior, BD_BOUNDS)

    # compute fitness
    fitness = list_l2_norm(behavior, [0, 0])

    env.close()
    return (behavior, (fitness,))


controllers_dict = {'discrete keypoints': controllers.ControllerDiscreteKeyPoints,
                    'interpolate keypoints': controllers.ControllerInterpolateKeyPoints}

bd_dict = {'2D': two_d_behavioral_descriptor,
           '3D': three_d_behavioral_descriptor}

if __name__ == "__main__":

    initial_genotype_size = NB_KEYPOINTS * 4
    evaluation_function = bd_dict[BD]
    pop, archive, hof = noveltysearch.novelty_algo(evaluation_function, initial_genotype_size, BD_BOUNDS,
                                                   mini=MINI,
                                                   plot=PLOT, algo_type=ALGO, nb_gen=30, bound_genotype=1,
                                                   pop_size=100, parallelize=PARALLELIZE, measures=True)

    hof_fit = np.array([ind.fitness.values for ind in hof])
    archive_fit = np.array([ind.fitness.values for ind in archive])

    if PLOT:
        # plot final states
        archive_behavior = np.array([ind.behavior_descriptor.values for ind in archive])
        pop_behavior = np.array([ind.behavior_descriptor.values for ind in pop])
        hof_behavior = np.array([ind.behavior_descriptor.values for ind in hof])
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set(title='Final Archive', xlabel='x1', ylabel='x2')
        ax.scatter(archive_behavior[:, 0], archive_behavior[:, 1], color='red', label='Archive')
        ax.scatter(pop_behavior[:, 0], pop_behavior[:, 1], color='blue', label='Population')
        ax.scatter(hof_behavior[:, 0], hof_behavior[:, 1], color='green', label='Hall of Fame')
        plt.legend()

    plt.show()
    
    if DISPLAY_HOF:
        DISPLAY = True
        for ind in hof:
            evaluate_individual(ind)
    
    if DISPLAY_RAND:
        DISPLAY = True
        nb_show = 10
        i = 0
        for ind in pop:
            if i <= nb_show:
                evaluate_individual(ind)
            else:
                break
            i += 1
