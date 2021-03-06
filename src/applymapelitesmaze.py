import gym
import gym_fastsim  # must still be imported
import mapelites
import time
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator

DISPLAY = False
PARALLELIZE = False


if PARALLELIZE:
    # container for behavior descriptor
    creator.create('BehaviorDescriptor', list)
    # container for info
    creator.create('Info', dict)
    # container for novelty
    creator.create('Novelty', base.Fitness, weights=(1.0,))
    # container for fitness
    if min:
        creator.create('Fit', base.Fitness, weights=(-1.0,))
    else:
        creator.create('Fit', base.Fitness, weights=(1.0,))

    # container for individual
    creator.create('Individual', list, behavior_descriptor=creator.BehaviorDescriptor,
                   novelty=creator.Novelty, fitness=creator.Fit, info=creator.Info)

    # set creator
    mapelites.set_creator(creator)


def evaluate_individual(individual):
    """Evaluates an individual: computes its value in the behavior descriptor space,
    and its fitness value.

    Args:
        individual (Individual): an individual

    Returns:
        tuple: tuple of behavior (list) and fitness(tuple)
    """
    inf = {}
    env = gym.make('FastsimSimpleNavigation-v0')
    env.reset()
    if(DISPLAY):
        env.enable_display()

    action = [0, 0]

    for i in range(1500):
        env.render()
        # apply previously chosen action
        o, r, eo, info = env.step(action)
        # normalize observations from sensor
        o[0] = o[0] / env.maxSensorRange
        o[1] = o[1] / env.maxSensorRange
        o[2] = o[2] / env.maxSensorRange

        # CONTROLLER
        # 5 inputs, 2 outputs
        individual = np.array(individual)
        o = np.array(o)
        action[0] = np.sum(np.multiply(individual[0:5], o)) + individual[5]
        action[1] = np.sum(np.multiply(individual[6:11], o)) + individual[11]

        # print("Step %d Obs=%s  reward=%f  dist. to objective=%f  robot position=%s  End of ep=%s" % (i, str(o), r,
        #     info["dist_obj"], str(info["robot_pos"]), str(eo)))
        if(DISPLAY):
            time.sleep(0.01)
        if eo:
            break
    # use last info to compute behavior and fitness
    behavior = [info["robot_pos"][0], info["robot_pos"][1]]
    fitness = info["dist_obj"]
    return (behavior, (fitness,), inf)


if __name__ == "__main__":
    plot = True
    initial_genotype_size = 12
    bd_bounds = [[0, 600], [0, 600]]
    grid, info = mapelites.map_elites_algo(evaluate_individual, initial_genotype_size, bd_bounds,
                                           mini=True, plot=plot, nb_gen=5000, parallelize=PARALLELIZE)

    if plot:
        # plot final states
        env = gym.make('FastsimSimpleNavigation-v0')
        env.reset()
        maze = env.map.get_data()
        maze = np.array([str(pix) for pix in maze])
        maze[maze == 'status_t.obstacle'] = 0.0
        maze[maze == 'status_t.free'] = 1.0
        maze = np.reshape(maze, (200, 200))
        maze = np.array(maze, dtype='float')
        grid_behavior = np.array([ind.behavior_descriptor.values for ind in grid if ind is not None])

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set(title='Final behaviors', xlabel='x1', ylabel='x2')
        ax.imshow(maze)
        ax.scatter(grid_behavior[:, 0] / 3, grid_behavior[:, 1] / 3, color='red')
        
    plt.show()
