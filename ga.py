from mchgenalg import GeneticAlgorithm
import mchgenalg
import numpy as np
import os

timesEvaluated = 0

# First, define function that will be used to evaluate the fitness
def fitness_function(genome):

    global timesEvaluated
    timesEvaluated += 1

    print("Fitness function invoked "+str(timesEvaluated)+" times")

    #setting parameter values using genome
    polyak = decode_function(genome[0:10])
    if polyak > 1:
        polyak = 1
    gamma = decode_function(genome[11:21])
    if gamma > 1:
        gamma = 1
    Q_lr = decode_function(genome[22:33])
    if Q_lr > 1:
        Q_lr = 1
    pi_lr = decode_function(genome[34:44])
    if pi_lr > 1:
        pi_lr = 1
    random_eps = decode_function(genome[45:55])
    if random_eps > 1:
        random_eps = 1
    epochs_default = 70 #80
    env = 'FetchPush-v1'
    logdir ='/tmp/openaiGA'
    num_cpu = 4

    query = "python3 -m baselines.her.experiment.train --env="+env+" --logdir="+logdir+" --n_epochs="+str(epochs_default)+" --num_cpu="+str(num_cpu) + " --polyak_value="+ str(polyak) + " --gamma_value=" + str(gamma) + " --Q_learning=" + str(Q_lr) + " --pi_learning" + str(pi_lr) + " --random_epsilon" + str(random_eps)
 + 
    #calling training to calculate number of epochs required to reach close to maximum success rate
    os.system(query)
    #epochs = train.launch(env, logdir, epochs_default, num_cpu, 0, 'future', 5, 1, polyak, gamma)
    #env, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return   

    file = open('epochs.txt', 'r')

    #one run is expected to converge before epochs_efault
    #if it does not converge, either add condition here, or make number of epochs as dynamic

    epochs = int(file.read())

    if epochs == None:
        epochs = epochs_default

    print('EPOCHS taken to converge:')
    print(epochs)
    return 1/epochs

def decode_function(genome_partial):

    prod = 0
    for i,e in reversed(list(enumerate(genome_partial))):
        if e == False:
            prod += 0
        else:
            prod += 2**abs(i-len(genome_partial)+1)
    return prod/1000

# Configure the algorithm:
population_size = 10
genome_length = 55
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

# Run 100 iteration of the algorithm
# You can call the method several times and adjust some parameters
# (e.g. number_of_pairs, selective_pressure, mutation_rate,
# allow_random_parent, single_point_cross_over)
ga.run(10)

best_genome, best_fitness = ga.get_best_genome()

print("BEST CHROMOSOME IS")
print(best_genome)
print("It's decoded value is")
print("Tau = " + decode_function(best_genome[0:10]))
print("Gamma = " + decode_function(best_genome[11:22]))

# If you want, you can have a look at the population:
population = ga.population

# and the fitness of each element:
fitness_vector = ga.get_fitness_vector()
