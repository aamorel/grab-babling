# Exploration of the basic tools of DEAP using the classical One Max problem
# Goal is to evolve the population towards having ones on all their dimensions

import matplotlib.pyplot as plt
import numpy as np
import random

from deap import base
from deap import creator
from deap import tools

CXPB = 0.5 # probability with which two individuals are crossed
MUTPB = 0.2 # probability for mutating an individual

# create container named FitnessMax, inheriting base.Fitness, with attribute weights
creator.create('FitnessMax', base.Fitness, weights=(1.0,))
# create container named Individual, inherting list, with attribute fitness
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox() # deap container

# register(alias, method[, argument[, ...]])
# Register a function in the toolbox under the name alias. You may provide default
# arguments that will be passed automatically when calling the registered function.
# Fixed arguments can then be overridden at function call time.

# register a generation function, the arguments following the definition
# of the function are always fed to the function
# function is accesible with toolbox.attr_bool
# format: toolbox.register(name, definition, arguments)
toolbox.register('attr_bool', random.randint, 0, 1)


# register initialization functions
# initRepeat(container, func, n)
# Call the function func n times and return the results in a container type container
toolbox.register('individual', tools.initRepeat, creator.Individual, \
    toolbox.attr_bool, 100)
# individual() will return an individual initialized with what would be returned
# by calling the attr_bool() method 100 times --> 100 dimensions in one individual
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    return sum(individual),

toolbox.register('evaluate', evalOneMax)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
toolbox.register('select', tools.selTournament, tournsize=3)

def gen_plot(mean_hist, max_hist):
    mean_hist = np.array(mean_hist)
    max_hist = np.array(max_hist)

    # plot evolution
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set(title='One Max problem', xlabel='Generations', ylabel='Fitness')
    ax.plot(mean_hist, label='Mean')
    ax.plot(max_hist, label='Max')
    plt.legend()
    plt.show()



def main():
    # pop is a list containing 300 individuals
    pop = toolbox.population(n=300)

    # evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop)) # same order in pop and fitnesses
    for ind, fit, in zip(pop, fitnesses):
        ind.fitness.values = fit # remember ind is a container with fitness as attribute

    # extracting all the fitnesses
    fits = [ind.fitness.values[0] for ind in pop]

    # keep track of number of generations
    gen = 0

    # keep track of stats
    mean_hist = []
    max_hist = []

    # begin the evolution
    while max(fits) < 100 and gen < 1000:
        # a new generation
        gen += 1
        print('--Generation %i --' % gen)

        offsprings = toolbox.select(pop, len(pop)) # list contains references to selected individuals
        offsprings = list(map(toolbox.clone, offsprings)) # clone selected indivduals
        for child1, child2 in zip(offsprings[::2], offsprings[1::2]):
            # child 1 has even index, child 2 has odd index
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                # both child1 and child2 are modified in place
                # del invalidates their fitness (new indivduals so unknown fitness)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offsprings:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # evaluate the indivduals with an invalid fitness
        invalid_ind = [ind for ind in offsprings if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # full replacement of population
        pop[:] = offsprings

        # gather all the fitnesses in one list and compute stats
        fits = np.array([ind.fitness.values[0] for ind in pop])
        mean = np.mean(fits)
        std = np.std(fits)
        print("  Min %s" % np.min(fits))
        print("  Max %s" % np.max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        mean_hist.append(mean)
        max_hist.append(np.max(fits))

    gen_plot(mean_hist, max_hist)


if __name__ == '__main__':
    main()
