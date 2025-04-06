from utils import calculate_fitness, initialize_population, mutate

def hill_climbing(data, max_iterations=500, neighbors_per_iter=20):
    current = initialize_population(data, 1)[0]
    current_fitness = calculate_fitness(current, data)

    for _ in range(max_iterations):
        best_neighbor = None
        best_fitness = float('-inf')

        for _ in range(neighbors_per_iter):
            neighbor = mutate(current, data, mutation_rate=0.2)
            fitness = calculate_fitness(neighbor, data)
            if fitness > best_fitness:
                best_fitness = fitness
                best_neighbor = neighbor

        if best_fitness > current_fitness:
            current = best_neighbor
            current_fitness = best_fitness
        else:
            break

    return current