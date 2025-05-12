import random
import math
from utils import calculate_fitness, initialize_population, mutate

# Algoritmo que tal como os outros procura o melhor schedule possivel
# Este tambem começa com soluções aleatorias

def simulated_annealing(data, initial_temp=1000.0, final_temp=0.1, alpha=0.95, max_iter_per_temp=100):
    current = initialize_population(data, 1)[0]
    current_fitness = calculate_fitness(current, data)
    # Usamos para guardar a melhor solução até ao momento
    best = current
    best_fitness = current_fitness
    # Inicializamos a temp
    temperature = initial_temp


    # Ciclo até a temperatura atingir o valor minimo
    while temperature > final_temp:
        # Várias iterações para cada temperatura
        for _ in range(max_iter_per_temp):
            neighbor = mutate(current, data, mutation_rate=0.2)
            # Mais uma vez calculamos o fitness de cada neighbour para ir comparando
            neighbor_fitness = calculate_fitness(neighbor, data)
            delta = neighbor_fitness - current_fitness

            # Aqui funciona como estados, em que decide se aceita ou não com base no delta
             
            if delta > 0 or random.random() < math.exp(delta / temperature):
                current = neighbor
                current_fitness = neighbor_fitness

                if current_fitness > best_fitness:
                    best = current
                    best_fitness = current_fitness

        # Redução da temperatura
        temperature *= alpha

    return best