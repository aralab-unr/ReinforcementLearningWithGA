from mchgenalg import GeneticAlgorithm
import numpy as np
import baselines.her.experiment.train as train

# First, define function that will be used to evaluate the fitness
def fitness_function(genome):
    # let's count the number of one-values in the genome
    # this will be our fitness
    #sum = np.sum(genome)

    print('String being tested is')
    print(genome)

    print('Decoded string is')
    print(decode_function(genome))

    #setting parameter values using genome
    polyak = 0.7 
    gamma = 0.98
    epochs_default = 70
    env = 'FetchPickAndPlace-v1'
    logdir = '/tmp/openaitest1'

    #calling training to calculate number of epochs required to reach close to maximum success rate
    epochs = train.launch(env, logdir, epochs_default, 1, 0, 'future', 5, 1, polyak, gamma)
    #env, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return

    return epochs

def decode_function(genome_partial):

    prod = 0
    for i in reversed(list(enumerate(genome_partial))):
        if genome_partial[i] == false:
            prod += 0
        else:
            prod += genome_partial[i] * 2**abs(i[0]-len(genome_partial)+1)
    return prod

# Configure the algorithm:
population_size = 10
genome_length = 2
ga = GeneticAlgorithm(fitness_function)
ga.generate_binary_population(size=population_size, genome_length=genome_length)
# How many pairs of individuals should be picked to mate
ga.number_of_pairs = 5
# Selective pressure from interval [1.0, 2.0]
# the lower value, the less will the fitness play role
ga.selective_pressure = 1.5
ga.mutation_rate = 0.1
# If two parents have the same genotype, ignore them and generate TWO random parents
# This helps preventing premature convergence
ga.allow_random_parent = True # default True
# Use single point crossover instead of uniform crossover
ga.single_point_cross_over = False # default False

# Run 1000 iteration of the algorithm
# You can call the method several times and adjust some parameters
# (e.g. number_of_pairs, selective_pressure, mutation_rate,
# allow_random_parent, single_point_cross_over)
ga.run(1000)

best_genome, best_fitness = ga.get_best_genome()

#print(best_fitness)

# If you want, you can have a look at the population:
population = ga.population
#print(population)

# and the fitness of each element:
fitness_vector = ga.get_fitness_vector()
#print(fitness_vector)
