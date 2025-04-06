from collections import deque
from utils import calculate_fitness, initialize_population, mutate


# Este algoritmo mantém um registo das ultimas soluções visitadas para evitar repetições e explorar novas áreas
# O objetivo é o mesmo, retornar o melhor schedule possivel

def tabu_search(data, max_iterations=500, tabu_size=50, neighbors_per_iter=30):
    current = initialize_population(data, 1)[0]
    current_fitness = calculate_fitness(current, data)
    best = current
    best_fitness = current_fitness

    # Inicializa a lista de soluções, como uma fila
    tabu_list = deque(maxlen=tabu_size)

    for iteration in range(max_iterations):
        neighbors = []
        
        for _ in range(neighbors_per_iter):
            neighbor = mutate(current, data, mutation_rate=0.2)
            # Criamos um hash para comparar com a lista de soluções 
            neighbor_hash = hash(frozenset(neighbor.items()))
            
            # Apenas vai considerar neighbours que nao estejam na lista
            if neighbor_hash not in tabu_list:
                fitness = calculate_fitness(neighbor, data)
                neighbors.append((neighbor, fitness, neighbor_hash))

        if not neighbors:
            break

        # Escolhe o melhor estado que não está na fila/lista    
        neighbor, neighbor_fitness, neighbor_hash = max(neighbors, key=lambda x: x[1])
        # Move-se para esse estado
        current = neighbor
        current_fitness = neighbor_fitness
        # Adicona à lista de soluções o hash do estado visitado
        tabu_list.append(neighbor_hash)


        # Atualiza a melhor solução
        if current_fitness > best_fitness:
            best = current
            best_fitness = current_fitness

    return best