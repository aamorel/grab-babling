import gym
import noveltysearch
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from deap import base, creator

DISPLAY = False
PARALLELIZE = True

NB_KEYPOINTS = 3
NB_ITER = 5000

if PARALLELIZE:
    # container for behavior descriptor
    creator.create('BehaviorDescriptor', list)
    # container for novelty
    creator.create('Novelty', base.Fitness, weights=(1.0,))
    # container for fitness
    if min:
        creator.create('Fit', base.Fitness, weights=(-1.0,))
    else:
        creator.create('Fit', base.Fitness, weights=(1.0,))

    # container for individual
    creator.create('Individual', list, behavior_descriptor=creator.BehaviorDescriptor,
                   novelty=creator.Novelty, fitness=creator.Fit)

    # set creator
    noveltysearch.set_creator(creator)


def choose_action(i):
    """Share the keypoints time equally

    Args:
        i (int): iteration index

    Returns:
        int: action index
    """
    nb_iter_per_action = math.ceil(NB_ITER / NB_KEYPOINTS)
    action_index = i // nb_iter_per_action
    return action_index


def list_l2_norm(list1, list2):
    if len(list1) != len(list2):
        raise NameError('The two lists have different length')
    else:
        dist = 0
        for i in range(len(list1)):
            dist += (list1[i] - list2[i]) ** 2
        dist = dist ** (1 / 2)
        return dist


def evaluate_individual(individual):
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
    print("Evaluation's process id is: ", os.getpid())
    env = gym.make('gym_baxter_grabbing:baxter_grabbing-v0', display=DISPLAY)
    env.reset()

    actions = []
    for i in range(NB_KEYPOINTS):
        actions.append(individual[3 * i:3 * (i + 1)])

    action = actions[0]

    for i in range(NB_ITER):
        env.render()
        # apply previously chosen action
        o, r, eo, info = env.step(action)

        if i == 0:
            initial_object_position = o[0]

        action = actions[choose_action(i)]

        if eo:
            break
    # use last info to compute behavior and fitness
    behavior = o[0]  # last position of object
    fitness = list_l2_norm(behavior, initial_object_position)
    env.close()
    return (behavior, (fitness,))


if __name__ == "__main__":
    plot = True
    initial_genotype_size = NB_KEYPOINTS * 3
    algo = 'classic_ea'

    pop, archive, hof = noveltysearch.novelty_algo(evaluate_individual, initial_genotype_size, min=True,
                                                   plot=plot, algo_type=algo, nb_gen=5, bound_genotype=1,
                                                   parallelize=PARALLELIZE)

    if plot:
        # plot final states
        archive_behavior = np.array([ind.behavior_descriptor.values for ind in archive])
        pop_behavior = np.array([ind.behavior_descriptor.values for ind in pop])
        hof_behavior = np.array([ind.behavior_descriptor.values for ind in hof])
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set(title='Final Archive', xlabel='x1', ylabel='x2')
        ax.scatter(archive_behavior[:, 0] / 3, archive_behavior[:, 1] / 3, color='red', label='Archive')
        ax.scatter(pop_behavior[:, 0] / 3, pop_behavior[:, 1] / 3, color='blue', label='Population')
        ax.scatter(hof_behavior[:, 0] / 3, hof_behavior[:, 1] / 3, color='green', label='Hall of Fame')
        plt.legend()

    plt.show()
