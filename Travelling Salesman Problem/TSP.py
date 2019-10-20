#importing necessary libraries
import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('TSP.csv')
distances = data.iloc[:, 1:].values
distances.shape
data.head()
print(distances)

#Creating Initial Population
num_generations = 500
solutions_per_pop = 8
pop_size = (solutions_per_pop, distances.shape[1])
initial_pop = np.empty(pop_size)
for i in range(pop_size[0]):
    initial_pop[i, :] = (rd.sample(range(1,16), pop_size[1]))
initial_pop = initial_pop.astype(int)
print('Initial Population: \n {0}'.format(initial_pop))
print(initial_pop.shape)
print(distances.shape)

def cal_fitness(population, distances):
    fitness = np.empty((len(population), 1))
    for i in range(len(population)):
        cost = 0
        for j in range(population.shape[1] - 1):
            city_1 = population[i][j]
            city_2 = population[i][j+1]
            cost = cost + distances[city_1 - 1][city_2 - 1]
        fitness[i][0]  = cost
    return fitness  
    
def selection(population, fitness, num_parents):
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        (individual_1, individual_2) = rd.sample(range(1,population.shape[0]+1), 2)
        if fitness[individual_1 - 1][0] < fitness[individual_2 - 1][0]:
            parent_idx = individual_1 - 1
        else:
            parent_idx = individual_2 - 1
        parents[i, :] = population[parent_idx, :]
    return parents  

def search(swath, element):
    for i in range(len(swath)):
        if swath[i] == element:
            return True
    return False

#Performing crossover using PMX
def crossover(parents, num_offsprings):
    offsprings = np.full((num_offsprings, parents.shape[1]), -999, dtype=int)
    i = 0 
    while i < parents.shape[0]:
        parent1 = parents[i, :]
        parent2 = parents[i+1, :]
        num_children = 1
        while num_children < 3:
            idx1 = rd.randint(0, parents.shape[1] - 2)
            idx2 = rd.randint(idx1+1, parents.shape[1] - 1)
            swath_p1 = parent1[idx1:idx2+1]
            swath_p2 = parent2[idx1:idx2+1]
            offsprings[i, idx1:idx2+1] = swath_p1

            for j in range(idx1, idx2+1, 1):
                p2_pos = p1_pos = j
                p2_element = parent2[j];
                if not search(swath_p1, p2_element):
                    flag = False
                    while not flag:
                        p1_element = parent1[p2_pos]
                        if search(swath_p2, p1_element):
                            p2_element = p1_element
                            p2_pos = np.where(parent2 == p1_element)
                            continue
                        flag = True    
                    offsprings[i, np.where(parent2 == p1_element)] = parent2[j]


            for j in range(offsprings.shape[1]):
                if offsprings[i, j] == -999:
                    offsprings[i, j] = parent2[j]
    
            parent1, parent2 = parent2, parent1
            i += 1
            num_children += 1
        
    return offsprings
    
def mutation(offsprings):
    mutation_rate = 0.40
    mutants = np.empty(offsprings.shape)
    
    for i in range(len(offsprings)):
        mutants[i, :] = offsprings[i, :]
        random_value = rd.random()
        if random_value > mutation_rate:
            continue
        idx1, idx2 = rd.sample(range(0, offsprings.shape[1]), 2)    
        mutants[i][idx1], mutants[i][idx2] = mutants[i][idx2], mutants[i][idx1]
        
    return mutants
    
def new_population(curr_population, distances, mutants, fitness_curr_pop):
    total_fitness = np.empty((len(curr_population) + len(mutants), 1))
    new_population = np.empty((curr_population.shape))
    fitness_mutants = cal_fitness(mutants, distances)
    total_fitness[0:len(curr_population), 0] = fitness_curr_pop[:, 0]
    total_fitness[len(curr_population): , 0] = fitness_mutants[:, 0]

    for i in range(len(new_population)):
        fittest_individual_idx = np.argmin(total_fitness)
        if fittest_individual_idx < len(curr_population):
            new_population[i, :] = curr_population[fittest_individual_idx, :]
        else:
            new_population[i, :] = mutants[fittest_individual_idx - len(curr_population), :]
        total_fitness[fittest_individual_idx] = 99999999
        
    return new_population        
    
def genetic_algorithm(population, distances, pop_size, num_generations):
    fitness_history, fittest_individual = [], []
    num_offsprings = num_parents = len(population)
    
    for i in range(num_generations):
        fitness = cal_fitness(population, distances)
        fitness_history.append(fitness)
        parents = selection(population, fitness, num_parents)
        offsprings = crossover(parents, num_offsprings)
        mutants = mutation(offsprings)
        mutants = mutants.astype(int)
        population = new_population(population, distances, mutants, fitness)
        population = population.astype(int)
    
    print('Last generation: \n{}\n'.format(population)) 
    fitness_last_gen = cal_fitness(population, distances)
    print('Fitness of the last generation: \n{}\n'.format(fitness_last_gen.astype(int)))
    max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    fittest_individual.append(population[max_fitness[0][0],:])
    return fitness_history, fittest_individual
    
fitness_history, calculated_path = genetic_algorithm(initial_pop, distances, pop_size, num_generations)
print('The path to be taken by salesman as calculated by Genetic Algorithm:\n{}'.format(list(calculated_path)))

fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
fitness_history_max = [np.max(fitness) for fitness in fitness_history]
plt.plot(list(range(num_generations)), fitness_history_mean, label = 'Mean Fitness')
plt.plot(list(range(num_generations)), fitness_history_max, label = 'Max Fitness')
plt.legend()
plt.title('Fitness through the generations')
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.show()
    

