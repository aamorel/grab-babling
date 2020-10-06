import random
import numpy as np
import math
import matplotlib.pyplot as plt
from deap import tools, base, algorithms, creator
from scipy.spatial import cKDTree as KDTree


INDIVIDUAL_SIZE = 2
POP_SIZE = 100
K = 5 # number of nearest neighbours for novelty computation
ARCHIVE = 'random'  # random or elitist
ARCHIVE_PB = 0.5 # if ARCHIVE is random, probability to add individual to archive

CXPB = 0.5 # probability with which two individuals are crossed
MUTPB = 0.2 # probability for mutating an individual
ABS_BOUND = 5 # absolute boundary for each dimension of individual genotype, can be None

def initialize_tool():
    # initialize the toolbox

    # container for behavior descriptor
    creator.create('BehaviorDescriptor', list)
    # container for novelty
    creator.create('Novelty', base.Fitness, weights=(1.0,))
    # container for fitness
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    # container for individual
    creator.create('Individual', list, behavior_descriptor=creator.BehaviorDescriptor,
        novelty=creator.Novelty, fitness=creator.FitnessMin)

    # create toolbox
    toolbox = base.Toolbox()

    # create function for individual initialization
    toolbox.register('init_ind', random.random)

    # create function for individual creation
    toolbox.register('individual', tools.initRepeat, creator.Individual, \
        toolbox.init_ind, INDIVIDUAL_SIZE)

    # create function for population creation
    toolbox.register('population', tools.initRepeat, list, toolbox.individual, POP_SIZE)

    # create operators
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=0.1, indpb=0.3)
    toolbox.register('select', tools.selTournament, tournsize=3)
    return creator, toolbox

def assess_behavior(individual):
    # returns the behavior of the individual in the behavior space
    # assumption: behavior is equal to genotype
    behavior = individual.copy()
    return behavior

def evaluate_fitness(individual):
    # returns the fitness of the individual as a tuple
    # example: Rastrigin function
    dim = len(individual)
    A = 10
    fitness = 0
    for i in range(dim):
        fitness += individual[i]**2 - A * math.cos(2 * math.pi * individual[i])
    fitness += A * dim
    return fitness,

def compute_average_distance(query, k_tree):
    # find K nearest neighbours and distances
    neighbours = k_tree.query(query, range(2, K + 2))
    # compute mean distance
    avg_distance = np.mean(neighbours[0])
    return avg_distance,

def assess_novelties(pop, archive):
    if not archive:
        # archive is empty --> only consider current population
        reference_pop = pop
    else:
        reference_pop = pop + archive
    # empty list for novelties
    novelties = []
    # extract all the behavior descriptors
    b_descriptors = [ind.behavior_descriptor.values for ind in reference_pop]
    k_tree = KDTree(b_descriptors)
    # compute novelty for current individuals
    for i in range(POP_SIZE):
        novelties.append(compute_average_distance(b_descriptors[i], k_tree))
    return novelties

def bound(offsprings):
    for ind in offsprings:
        for i in range(len(ind)):
            if ind[i] > ABS_BOUND:
                ind[i] = ABS_BOUND
            if ind[i] < -ABS_BOUND:
                ind[i] = ABS_BOUND

def operate_offsprings(offsprings, toolbox):
    # crossover
    for child1, child2 in zip(offsprings[::2], offsprings[1::2]):
        # child 1 has even index, child 2 has odd index
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            # both child1 and child2 are modified in place
            # del invalidates their fitness (new indivduals so unknown fitness)
            del child1.fitness.values
            del child2.fitness.values
    # mutation
    for mutant in offsprings:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # bound genotype to given constraints
    if ABS_BOUND is not None:
        bound(offsprings)

def gen_plot(mean_hist, min_hist, arch_size_hist):
    mean_hist = np.array(mean_hist)
    min_hist = np.array(min_hist)

    # plot evolution
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set(title='Test with Rastrigin function', xlabel='Generations', ylabel='Fitness')
    ax.plot(mean_hist, label='Mean')
    ax.plot(min_hist, label='Min')
    plt.legend()

    # plot evolution
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set(title='Evolution of archive size', xlabel='Generations', ylabel='Archive size')
    ax.plot(arch_size_hist)


def main():

    creator, toolbox = initialize_tool()
    pop = toolbox.population()

    # plot initial population
    pop_n = np.array(pop)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set(title='Initial Population', xlabel='x1', ylabel='x2')
    ax.scatter(pop_n[:,0], pop_n[:,1])
    for i in range(len(pop)):
        ax.annotate(i, (pop_n[i,0], pop_n[i,1]))

    # initialize the archive
    archive = []

    # initialize generation counter
    gen = 0

    # evaluate initial population
    b_descriptors = list(map(assess_behavior, pop))
    fitnesses = list(map(evaluate_fitness, pop))

    # attribute fitness and behavior descriptors to individuals
    for ind, fit, bd in zip(pop, fitnesses, b_descriptors):
        ind.behavior_descriptor.values = bd
        ind.fitness.values = fit

    novelties = assess_novelties(pop, archive)
    for ind, nov in zip(pop, novelties):
        ind.novelty.values = nov

    # fill archive
    if ARCHIVE == 'random':
        # fill archive randomly
        for ind in pop:
            if random.random() < ARCHIVE_PB:
                member = toolbox.clone(ind)
                archive.append(member)
    # TODO: if ARCHIVE == 'elitist':

    #keep track of stats
    mean_hist = []
    min_hist = []
    arch_size_hist = []

    # begin evolution
    while gen < 100:
        # a new generation
        gen += 1
        print('--Generation %i --' % gen)

        # novelty search: selection on novelty
        offsprings = toolbox.select(pop, len(pop), fit_attr='novelty') # list contains references to selected individuals

        # classical EA: selection on fitness
        # offsprings = toolbox.select(pop, len(pop), fit_attr='fitness') # list contains references to selected individuals

        offsprings = list(map(toolbox.clone, offsprings)) # clone selected indivduals
        operate_offsprings(offsprings, toolbox) # crossover and mutation

        # evaluate the indivduals with an invalid fitness
        invalid_ind = [ind for ind in offsprings if not ind.fitness.valid]
        inv_b_descriptors = list(map(assess_behavior, invalid_ind))
        inv_fitnesses = list(map(evaluate_fitness, invalid_ind))

        # attribute fitness and behavior descriptors to new individuals
        for ind, fit, bd in zip(invalid_ind, inv_fitnesses, inv_b_descriptors):
            ind.behavior_descriptor.values = bd
            ind.fitness.values = fit

        # full replacement of population
        pop[:] = offsprings

        # compute novelty for all current individuals (also old ones)
        novelties = assess_novelties(pop, archive)
        for ind, nov in zip(pop, novelties):
            ind.novelty.values = nov

        # fill archive
        if ARCHIVE == 'random':
            # fill archive randomly
            for ind in pop:
                if random.random() < ARCHIVE_PB:
                    member = toolbox.clone(ind)
                    archive.append(member)

        # gather all the fitnesses in one list and compute stats
        fits = np.array([ind.fitness.values[0] for ind in pop])
        mean = np.mean(fits)
        std = np.std(fits)
        print("  Min %s" % np.min(fits))
        print("  Max %s" % np.max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        arch_size_hist.append(len(archive))
        mean_hist.append(mean)
        min_hist.append(np.min(fits))

    gen_plot(mean_hist, min_hist, arch_size_hist)

    # plot initial population
    archive_n = np.array(archive)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set(title='Final Archive', xlabel='x1', ylabel='x2')
    ax.scatter(archive_n[:,0], archive_n[:,1])

    # show all plots
    plt.show()

if __name__ == '__main__':
    main()
