from collections import deque
from utils import calculate_fitness, initialize_population, mutate

def tabu_search(data, max_iterations=500, tabu_size=50, neighbors_per_iter=30):
    current = initialize_population(data, 1)[0]
    current_fitness = calculate_fitness(current, data)
    best = current
    best_fitness = current_fitness

    tabu_list = deque(maxlen=tabu_size)

    for iteration in range(max_iterations):
        neighbors = []
        
        for _ in range(neighbors_per_iter):
            neighbor = mutate(current, data, mutation_rate=0.2)
            neighbor_hash = hash(frozenset(neighbor.items()))
            
            if neighbor_hash not in tabu_list:
                fitness = calculate_fitness(neighbor, data)
                neighbors.append((neighbor, fitness, neighbor_hash))

        if not neighbors:
            break

        neighbor, neighbor_fitness, neighbor_hash = max(neighbors, key=lambda x: x[1])

        current = neighbor
        current_fitness = neighbor_fitness
        tabu_list.append(neighbor_hash)

        if current_fitness > best_fitness:
            best = current
            best_fitness = current_fitness

    return best