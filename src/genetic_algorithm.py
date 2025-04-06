
import random
from utils import calculate_fitness, initialize_population, mutate

def crossover(parent1, parent2, data):
    child = {}
    for patient in data["patients"]:
        pid = patient["patient_id"]
        if random.random() < 0.5:
            child[pid] = parent1[pid]
        else:
            child[pid] = parent2[pid]
    return child

def genetic_algorithm(data, population_size=50, generations=100, mutation_rate=0.1):
    population = initialize_population(data, population_size)
    
    for generation in range(generations):
        population = sorted(
            population,
            key=lambda s: calculate_fitness(s, data),
            reverse=True
        )
        
        elites = population[:int(0.2 * population_size)]
        new_population = elites.copy()
        
        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population[:10], k=2)  
            child = crossover(parent1, parent2, data)
            child = mutate(child, data, mutation_rate)
            new_population.append(child)
        
        population = new_population
    
    return max(population, key=lambda s: calculate_fitness(s, data))