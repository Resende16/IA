import random
import math
from utils import calculate_fitness, initialize_population, mutate

def simulated_annealing(data, initial_temp=1000.0, final_temp=0.1, alpha=0.95, max_iter_per_temp=100):
    current = initialize_population(data, 1)[0]
    current_fitness = calculate_fitness(current, data)
    best = current
    best_fitness = current_fitness
    temperature = initial_temp

    while temperature > final_temp:
        for _ in range(max_iter_per_temp):
            neighbor = mutate(current, data, mutation_rate=0.2)
            neighbor_fitness = calculate_fitness(neighbor, data)
            delta = neighbor_fitness - current_fitness

            if delta > 0 or random.random() < math.exp(delta / temperature):
                current = neighbor
                current_fitness = neighbor_fitness

                if current_fitness > best_fitness:
                    best = current
                    best_fitness = current_fitness

        temperature *= alpha

    return best