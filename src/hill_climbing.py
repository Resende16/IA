from utils import calculate_fitness, initialize_population, mutate


# Tal como o genetic, vai retornar o melhor calendario possivel, sempre tendo em conta o fitness
# O algortimo começa com uma solução aleatoria e evolui para solucoes cada vez melhores, tal como o nome "hill climbing" mostra


def hill_climbing(data, max_iterations=500, neighbors_per_iter=20):

    # Solução inicial aleatoria
    current = initialize_population(data, 1)[0]
    current_fitness = calculate_fitness(current, data)


    for _ in range(max_iterations):
        best_neighbor = None
        best_fitness = float('-inf') # Inicializa com um numero muito baixo e a ideia é ir subindo e aproximar de zero

        for _ in range(neighbors_per_iter):
            neighbor = mutate(current, data, mutation_rate=0.2)

            # Calculamos constantemente o fitness do melhor vizinho
            fitness = calculate_fitness(neighbor, data)
            if fitness > best_fitness:
                best_fitness = fitness
                best_neighbor = neighbor

        if best_fitness > current_fitness:
            current = best_neighbor
            current_fitness = best_fitness
        else:
            break

        # Assim, comparando o fitness dos vizinhos, vai retornar sempre o que tem melhor (mais proximo de zero)

    return current