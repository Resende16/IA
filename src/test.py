
import random
import math
from rich.console import Console
from rich.table import Table
from collections import deque
from datetime import datetime
import re

console = Console()

#Parser Part ---------------------------------------------------------------------------------------------------------

def parse_instance_file(file_path):
    data = {
        "seed": None,
        "minor_specialisms_per_ward": None,
        "weights": {},
        "days": None,
        "specialisms": {},
        "wards": {},
        "patients": []
    }
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    index = 0
    data["seed"] = int(re.findall(r'\d+', lines[index])[0])
    index += 1
    data["minor_specialisms_per_ward"] = int(re.findall(r'\d+', lines[index])[0])
    index += 1
    
    # Read weights
    data["weights"] = {
        "overtime": float(lines[index].split(': ')[-1]),
        "undertime": float(lines[index + 1].split(': ')[-1]),
        "delay": float(lines[index + 2].split(': ')[-1])
    }
    index += 3
    
    # Read days
    data["days"] = int(lines[index].split(': ')[-1])
    index += 1
    
    # Read specialisms
    num_specialisms = int(lines[index].split(': ')[-1])
    index += 1
    for _ in range(num_specialisms):
        parts = lines[index].strip().split()
        spec_id = parts[0]
        workload_factor = float(parts[1])
        ot_time = list(map(int, parts[2].split(';')))
        data["specialisms"][spec_id] = {"workload_factor": workload_factor, "ot_time": ot_time}
        index += 1
    
    # Read wards
    num_wards = int(lines[index].split(': ')[-1])
    index += 1
    for _ in range(num_wards):
        parts = lines[index].strip().split()
        ward_id = parts[0]
        bed_capacity = int(parts[1])
        workload_capacity = float(parts[2])
        major_specialization = parts[3]
        minor_specializations = parts[4] if parts[4] != "NONE" else []
        carryover_patients = list(map(int, parts[5].split(';')))
        carryover_workload = list(map(float, parts[6].split(';')))
        
        data["wards"][ward_id] = {
            "bed_capacity": bed_capacity,
            "workload_capacity": workload_capacity,
            "major_specialization": major_specialization,
            "minor_specializations": minor_specializations,
            "carryover_patients": carryover_patients,
            "carryover_workload": carryover_workload
        }
        index += 1
    
    # Read patients
    num_patients = int(lines[index].split(': ')[-1])
    index += 1
    for _ in range(num_patients):
        parts = lines[index].strip().split()
        patient_id = parts[0]
        specialization = parts[1]
        earliest_admission = int(parts[2])
        latest_admission = int(parts[3])
        length_of_stay = int(parts[4])
        surgery_duration = int(parts[5])
        workload_per_day = list(map(float, parts[6].split(';')))
        
        data["patients"].append({
            "patient_id": patient_id,
            "specialization": specialization,
            "earliest_admission": earliest_admission,
            "latest_admission": latest_admission,
            "length_of_stay": length_of_stay,
            "surgery_duration": surgery_duration,
            "workload_per_day": workload_per_day
        })
        index += 1
    
    return data

#End Parser -----------------------------------------------------------------------------------------------------------------------------

def calculate_fitness(schedule, data):
    """
    Calcula o fitness de uma agenda com base no custo total (menor é melhor).
    Retorna o negativo do custo para conversão em maximização.
    """
    total_cost = 0
    daily_workload = {ward: [0.0] * data["days"] for ward in data["wards"]}
    daily_beds_used = {ward: [0] * data["days"] for ward in data["wards"]}

    # Inicialize workload com pacientes carryover
    for ward in data["wards"]:
        for day in range(min(len(data["wards"][ward]["carryover_workload"]), data["days"])):
            daily_workload[ward][day] += data["wards"][ward]["carryover_workload"][day]
            daily_beds_used[ward][day] += data["wards"][ward]["carryover_patients"][day]

    for patient_id, (ward, admission_day, _) in schedule.items():
        patient = next(p for p in data["patients"] if p["patient_id"] == patient_id)
        
        # Penalidade para pacientes alocados a enfermarias sem especialização adequada
        if patient["specialization"] != data["wards"][ward]["major_specialization"]:
            total_cost += 1e6  # Penalidade severa
        
        # Acumular workload e uso de camas para cada dia da estadia
        for day_offset in range(patient["length_of_stay"]):
            current_day = admission_day + day_offset
            if current_day >= data["days"]:
                continue  # Ignorar dias além do horizonte de planejamento
            
            daily_workload[ward][current_day] += patient["workload_per_day"][day_offset]
            daily_beds_used[ward][current_day] += 1
            
        # Penalidade para atrasos de admissão
        if admission_day > patient["earliest_admission"]:
            total_cost += data["weights"]["delay"] * (admission_day - patient["earliest_admission"])

    # Verificar capacidade e calcular custos de horas extras
    for ward in data["wards"]:
        for day in range(data["days"]):
            # Penalidade para horas extras
            if daily_workload[ward][day] > data["wards"][ward]["workload_capacity"]:
                total_cost += data["weights"]["overtime"] * (
                    daily_workload[ward][day] - data["wards"][ward]["workload_capacity"]
                )
            
            # Penalidade para subaproveitamento
            elif daily_workload[ward][day] < data["wards"][ward]["workload_capacity"]:
                total_cost += data["weights"]["undertime"] * (
                    data["wards"][ward]["workload_capacity"] - daily_workload[ward][day]
                )
            
            # Penalidade severa para excesso de capacidade de camas
            if daily_beds_used[ward][day] > data["wards"][ward]["bed_capacity"]:
                total_cost += 1e6 * (daily_beds_used[ward][day] - data["wards"][ward]["bed_capacity"])

    return -total_cost  # Convertendo para problema de maximização

def create_initial_solution(data, strategy="random"):
    """
    Cria uma solução inicial usando diferentes estratégias
    """
    schedule = {}
    
    if strategy == "random":
        # Estratégia aleatória
        for patient in data["patients"]:
            valid_wards = [
                ward for ward in data["wards"] 
                if data["wards"][ward]["major_specialization"] == patient["specialization"]
            ]
            if not valid_wards:
                raise ValueError(f"No valid ward for patient {patient['patient_id']}")
            
            chosen_ward = random.choice(valid_wards)
            admission_day = random.randint(
                patient["earliest_admission"],
                min(patient["latest_admission"], data["days"] - 1)
            )
            schedule[patient["patient_id"]] = (chosen_ward, admission_day, patient["length_of_stay"])
    
    elif strategy == "earliest":
        # Estratégia de admissão mais cedo possível
        for patient in data["patients"]:
            valid_wards = [
                ward for ward in data["wards"] 
                if data["wards"][ward]["major_specialization"] == patient["specialization"]
            ]
            if not valid_wards:
                raise ValueError(f"No valid ward for patient {patient['patient_id']}")
            
            chosen_ward = random.choice(valid_wards)
            schedule[patient["patient_id"]] = (chosen_ward, patient["earliest_admission"], patient["length_of_stay"])
    
    elif strategy == "latest":
        # Estratégia de admissão mais tarde possível
        for patient in data["patients"]:
            valid_wards = [
                ward for ward in data["wards"] 
                if data["wards"][ward]["major_specialization"] == patient["specialization"]
            ]
            if not valid_wards:
                raise ValueError(f"No valid ward for patient {patient['patient_id']}")
            
            chosen_ward = random.choice(valid_wards)
            admission_day = min(patient["latest_admission"], data["days"] - patient["length_of_stay"])
            schedule[patient["patient_id"]] = (chosen_ward, admission_day, patient["length_of_stay"])
    
    elif strategy == "balanced":
        # Estratégia equilibrada por carga de trabalho
        ward_loads = {ward: [0] * data["days"] for ward in data["wards"]}
        
        # Ordenar pacientes por urgência (earliest_admission mais baixo)
        sorted_patients = sorted(data["patients"], key=lambda x: x["earliest_admission"])
        
        for patient in sorted_patients:
            valid_wards = [
                ward for ward in data["wards"] 
                if data["wards"][ward]["major_specialization"] == patient["specialization"]
            ]
            if not valid_wards:
                raise ValueError(f"No valid ward for patient {patient['patient_id']}")
            
            best_ward = None
            best_day = None
            min_load = float('inf')
            
            # Encontrar o melhor dia e enfermaria com menor carga
            for ward in valid_wards:
                for day in range(patient["earliest_admission"], min(patient["latest_admission"] + 1, data["days"])):
                    # Check if adding this patient would exceed bed capacity
                    if all(ward_loads[ward][d] < data["wards"][ward]["bed_capacity"] 
                           for d in range(day, min(day + patient["length_of_stay"], data["days"]))):
                        total_load = sum(ward_loads[ward][d] for d in range(day, min(day + patient["length_of_stay"], data["days"])))
                        if total_load < min_load:
                            min_load = total_load
                            best_ward = ward
                            best_day = day
            
            # If no valid placement was found, use earliest possible day
            if best_ward is None:
                best_ward = random.choice(valid_wards)
                best_day = patient["earliest_admission"]
            
            # Update load for the scheduled days
            for d in range(best_day, min(best_day + patient["length_of_stay"], data["days"])):
                ward_loads[best_ward][d] += 1
            
            schedule[patient["patient_id"]] = (best_ward, best_day, patient["length_of_stay"])
    
    return schedule

def initialize_population(data, population_size, strategies=None):
    """
    Inicializa uma população diversificada usando diferentes estratégias
    """
    if strategies is None:
        strategies = ["random", "earliest", "latest", "balanced"]
    
    population = []
    
    # Garante pelo menos uma solução de cada estratégia
    for strategy in strategies:
        population.append(create_initial_solution(data, strategy))
    
    # Completa o resto da população
    remaining = population_size - len(strategies)
    for _ in range(remaining):
        strategy = random.choice(strategies)
        solution = create_initial_solution(data, strategy)
        # Aplica uma mutação leve para diversificar
        solution = mutate(solution, data, mutation_rate=0.3)
        population.append(solution)
    
    return population

def mutate(schedule, data, mutation_rate=0.1, mutation_strength="normal"):
    """
    Muta uma agenda com diferentes estratégias de força
    """
    mutated = schedule.copy()
    
    if mutation_strength == "light":
        # Mutação leve: apenas modifica o dia de admissão
        for patient in data["patients"]:
            if random.random() < mutation_rate:
                pid = patient["patient_id"]
                ward, old_day, los = mutated[pid]
                
                # Escolher um novo dia próximo ao atual
                day_range = max(1, int((patient["latest_admission"] - patient["earliest_admission"]) * 0.2))
                new_day = max(patient["earliest_admission"], 
                             min(patient["latest_admission"], 
                                 old_day + random.randint(-day_range, day_range)))
                
                mutated[pid] = (ward, new_day, los)
    
    elif mutation_strength == "normal":
        # Mutação normal: modifica aleatoriamente dia ou enfermaria
        for patient in data["patients"]:
            if random.random() < mutation_rate:
                pid = patient["patient_id"]
                ward, day, los = mutated[pid]
                valid_wards = [
                    w for w in data["wards"] 
                    if data["wards"][w]["major_specialization"] == patient["specialization"]
                ]
                
                if random.random() < 0.5 and len(valid_wards) > 1:  # Mudar enfermaria
                    new_ward = random.choice([w for w in valid_wards if w != ward])
                    mutated[pid] = (new_ward, day, los)
                else:  # Mudar dia
                    new_day = random.randint(
                        patient["earliest_admission"],
                        min(patient["latest_admission"], data["days"] - 1)
                    )
                    mutated[pid] = (ward, new_day, los)
    
    elif mutation_strength == "strong":
        # Mutação forte: alta chance de mudar tanto o dia quanto a enfermaria
        for patient in data["patients"]:
            if random.random() < mutation_rate:
                pid = patient["patient_id"]
                valid_wards = [
                    ward for ward in data["wards"] 
                    if data["wards"][ward]["major_specialization"] == patient["specialization"]
                ]
                
                new_ward = random.choice(valid_wards)
                new_day = random.randint(
                    patient["earliest_admission"],
                    min(patient["latest_admission"], data["days"] - 1)
                )
                mutated[pid] = (new_ward, new_day, los)
    
    return mutated

def crossover(parent1, parent2, data, method="uniform"):
    """
    Combina dois pais para criar um filho usando diferentes métodos
    """
    child = {}
    
    if method == "uniform":
        # Crossover uniforme - para cada paciente, escolha aleatória de um dos pais
        for patient in data["patients"]:
            pid = patient["patient_id"]
            if random.random() < 0.5:
                child[pid] = parent1[pid]
            else:
                child[pid] = parent2[pid]
    
    elif method == "one_point":
        # Crossover de um ponto - escolhe um ponto e divide os pacientes entre os pais
        sorted_patients = sorted(data["patients"], key=lambda x: x["patient_id"])
        split_point = random.randint(1, len(sorted_patients) - 1)
        
        for i, patient in enumerate(sorted_patients):
            pid = patient["patient_id"]
            if i < split_point:
                child[pid] = parent1[pid]
            else:
                child[pid] = parent2[pid]
    
    elif method == "ward_based":
        # Crossover baseado em enfermarias - agrupa por enfermaria
        wards = list(data["wards"].keys())
        random.shuffle(wards)
        split_point = random.randint(1, len(wards) - 1)
        ward_set1 = set(wards[:split_point])
        
        for patient in data["patients"]:
            pid = patient["patient_id"]
            ward1, _, _ = parent1[pid]
            
            if ward1 in ward_set1:
                child[pid] = parent1[pid]
            else:
                child[pid] = parent2[pid]
    
    return child

def genetic_algorithm(data, population_size=50, generations=100, mutation_rate=0.1):
    """
    Algoritmo genético melhorado
    """
    # Inicializar população com diferentes estratégias
    population = initialize_population(data, population_size, 
                                     strategies=["random", "earliest", "latest", "balanced"])
    
    best_schedule = None
    best_fitness = float('-inf')
    
    console.print("[bold green]Running Genetic Algorithm...[/bold green]")
    
    # Parâmetros adaptativos
    crossover_methods = ["uniform", "one_point", "ward_based"]
    mutation_strengths = ["light", "normal", "strong"]
    elite_fraction = 0.2
    
    for generation in range(generations):
        # Avalia e ordena a população
        evaluated_population = [(s, calculate_fitness(s, data)) for s in population]
        evaluated_population.sort(key=lambda x: x[1], reverse=True)
        population = [s for s, _ in evaluated_population]
        fitness_values = [f for _, f in evaluated_population]
        
        # Acompanha a melhor solução
        if fitness_values[0] > best_fitness:
            best_fitness = fitness_values[0]
            best_schedule = population[0]
            
        # Imprime progresso a cada 10 gerações
        if generation % 10 == 0:
            console.print(f"Generation {generation}: Best fitness = {best_fitness}")
        
        # Adapta parâmetros com base no progresso
        if generation > 0 and generation % 20 == 0:
            diversity = len(set(fitness_values)) / len(fitness_values)
            if diversity < 0.1:  # Baixa diversidade
                mutation_rate = min(0.5, mutation_rate * 1.5)  # Aumenta mutação
                elite_fraction = max(0.1, elite_fraction * 0.8)  # Reduz elitismo
            else:
                mutation_rate = max(0.05, mutation_rate * 0.9)  # Reduz mutação
                elite_fraction = min(0.4, elite_fraction * 1.1)  # Aumenta elitismo
        
        # Seleção e elitismo
        num_elites = max(1, int(elite_fraction * population_size))
        elites = population[:num_elites]
        new_population = elites.copy()
        
        # Criação da nova população
        while len(new_population) < population_size:
            # Torneio para seleção de pais
            tournament_size = max(2, int(0.1 * population_size))
            candidates = random.sample(population, tournament_size)
            parent1 = max(candidates, key=lambda s: calculate_fitness(s, data))
            
            candidates = random.sample(population, tournament_size)
            parent2 = max(candidates, key=lambda s: calculate_fitness(s, data))
            
            # Crossover com método aleatório
            crossover_method = random.choice(crossover_methods)
            child = crossover(parent1, parent2, data, method=crossover_method)
            
            # Mutação com força adaptativa
            mutation_strength = random.choice(mutation_strengths)
            child = mutate(child, data, mutation_rate, mutation_strength)
            
            new_population.append(child)
        
        population = new_population
    
    console.print(f"[bold green]GA completed after {generations} generations.[/bold green]")
    console.print(f"Best fitness: {best_fitness}")
    
    return best_schedule

def get_neighbors(current, data, num_neighbors=10, neighborhood_strategy="mixed"):
    """
    Gera vizinhos para uma solução atual usando diferentes estratégias
    """
    neighbors = []
    
    if neighborhood_strategy == "patient_swap":
        # Estratégia de troca de pacientes
        patient_ids = list(current.keys())
        for _ in range(num_neighbors):
            if len(patient_ids) < 2:
                continue
                
            neighbor = current.copy()
            # Seleciona dois pacientes diferentes aleatoriamente
            p1, p2 = random.sample(patient_ids, 2)
            
            # Troca as alocações
            ward1, day1, los1 = neighbor[p1]
            ward2, day2, los2 = neighbor[p2]
            
            # Garante que as enfermarias são compatíveis com as especializações
            patient1 = next(p for p in data["patients"] if p["patient_id"] == p1)
            patient2 = next(p for p in data["patients"] if p["patient_id"] == p2)
            
            if (data["wards"][ward2]["major_specialization"] == patient1["specialization"] and
                data["wards"][ward1]["major_specialization"] == patient2["specialization"]):
                # Troca completa de alocação
                neighbor[p1] = (ward2, day1, los1)
                neighbor[p2] = (ward1, day2, los2)
                neighbors.append(neighbor)
    
    elif neighborhood_strategy == "day_shift":
        # Estratégia de deslocamento de dias
        for _ in range(num_neighbors):
            neighbor = current.copy()
            # Seleciona um paciente aleatório
            patient_id = random.choice(list(neighbor.keys()))
            patient = next(p for p in data["patients"] if p["patient_id"] == patient_id)
            
            ward, day, los = neighbor[patient_id]
            
            # Desloca o dia de admissão
            shift = random.choice([-2, -1, 1, 2])
            new_day = max(patient["earliest_admission"], 
                          min(patient["latest_admission"], day + shift))
            
            if new_day != day:
                neighbor[patient_id] = (ward, new_day, los)
                neighbors.append(neighbor)
    
    elif neighborhood_strategy == "ward_change":
        # Estratégia de mudança de enfermaria
        for _ in range(num_neighbors):
            neighbor = current.copy()
            # Seleciona um paciente aleatório
            patient_id = random.choice(list(neighbor.keys()))
            patient = next(p for p in data["patients"] if p["patient_id"] == patient_id)
            
            ward, day, los = neighbor[patient_id]
            
            # Encontra enfermarias válidas
            valid_wards = [
                w for w in data["wards"] 
                if data["wards"][w]["major_specialization"] == patient["specialization"] and w != ward
            ]
            
            if valid_wards:
                new_ward = random.choice(valid_wards)
                neighbor[patient_id] = (new_ward, day, los)
                neighbors.append(neighbor)
    
    elif neighborhood_strategy == "mixed":
        # Mistura de várias estratégias
        strategies = ["patient_swap", "day_shift", "ward_change"]
        num_per_strategy = max(1, num_neighbors // len(strategies))
        
        for strategy in strategies:
            strategy_neighbors = get_neighbors(current, data, num_per_strategy, strategy)
            neighbors.extend(strategy_neighbors)
    
    return neighbors

def hill_climbing(data, max_iterations=500, neighbors_per_iter=20):
    """
    Hill Climbing melhorado
    """
    # Inicializa com uma estratégia balanceada
    current = create_initial_solution(data, "balanced")
    current_fitness = calculate_fitness(current, data)
    best = current
    best_fitness = current_fitness
    
    console.print("[bold green]Running Hill Climbing...[/bold green]")
    
    # Estratégia de reinício
    restarts = 0
    max_restarts = 3
    plateau_counter = 0
    max_plateau = 50
    
    for iteration in range(max_iterations):
        # Gera vizinhos
        neighborhood_strategy = "mixed" if iteration % 3 == 0 else random.choice(["patient_swap", "day_shift", "ward_change"])
        neighbors = get_neighbors(current, data, neighbors_per_iter, neighborhood_strategy)
        
        if not neighbors:
            console.print("No valid neighbors found, generating through mutation")
            neighbors = [mutate(current, data, 0.3) for _ in range(neighbors_per_iter)]
        
        # Encontra o melhor vizinho
        best_neighbor = None
        best_neighbor_fitness = float('-inf')
        
        for neighbor in neighbors:
            fitness = calculate_fitness(neighbor, data)
            if fitness > best_neighbor_fitness:
                best_neighbor_fitness = fitness
                best_neighbor = neighbor
        
        # Atualiza se encontrou vizinho melhor
        if best_neighbor_fitness > current_fitness:
            current = best_neighbor
            current_fitness = best_neighbor_fitness
            plateau_counter = 0
            
            # Atualiza a melhor solução global
            if current_fitness > best_fitness:
                best = current
                best_fitness = current_fitness
                console.print(f"Iteration {iteration}: New best fitness = {best_fitness}")
        else:
            plateau_counter += 1
        
        # Reinício se estiver em um platô
        if plateau_counter >= max_plateau and restarts < max_restarts:
            console.print(f"Restarting after plateau of {plateau_counter} iterations")
            # Estratégia diferente a cada reinício
            strategies = ["random", "earliest", "latest", "balanced"]
            current = create_initial_solution(data, strategies[restarts % len(strategies)])
            current_fitness = calculate_fitness(current, data)
            plateau_counter = 0
            restarts += 1
    
    console.print(f"[bold green]Hill Climbing completed after {iteration+1} iterations with {restarts} restarts.[/bold green]")
    console.print(f"Best fitness: {best_fitness}")
    
    return best

def simulated_annealing(data, initial_temp=1000.0, final_temp=0.1, alpha=0.95, max_iter_per_temp=100):
    """
    Simulated Annealing melhorado
    """
    # Inicia com uma estratégia aleatória
    current = create_initial_solution(data, "random")
    current_fitness = calculate_fitness(current, data)
    best = current
    best_fitness = current_fitness
    temperature = initial_temp
    
    console.print("[bold green]Running Simulated Annealing...[/bold green]")
    
    total_iterations = 0
    accepted_moves = 0
    improved_moves = 0
    
    # Fator de reaquecimento adaptativo
    reheat_factor = 1.5
    stagnation_counter = 0
    stagnation_limit = 5
    
    while temperature > final_temp:
        moves_at_temp = 0
        accepted_at_temp = 0
        
        for _ in range(max_iter_per_temp):
            total_iterations += 1
            moves_at_temp += 1
            
            # Alternar entre estratégias de vizinhança
            neighborhood_strategy = "mixed" if temperature > 100 else random.choice(["day_shift", "ward_change"])
            neighbors = get_neighbors(current, data, 1, neighborhood_strategy)
            
            if not neighbors:
                # Se não encontrar vizinhos, usar mutação
                neighbor = mutate(current, data, 0.2, "normal")
            else:
                neighbor = neighbors[0]
                
            neighbor_fitness = calculate_fitness(neighbor, data)
            delta = neighbor_fitness - current_fitness
            
            # Critério de aceitação de Metropolis
            if delta > 0 or random.random() < math.exp(delta / temperature):
                current = neighbor
                current_fitness = neighbor_fitness
                accepted_moves += 1
                accepted_at_temp += 1
                
                if delta > 0:
                    improved_moves += 1
                    stagnation_counter = 0
                    
                    if current_fitness > best_fitness:
                        best = current
                        best_fitness = current_fitness
                        console.print(f"Temperature {temperature:.2f}, Iteration {total_iterations}: New best fitness = {best_fitness}")
            
            # Monitoramento de desempenho
            if total_iterations % 100 == 0:
                acceptance_rate = accepted_moves / total_iterations
                improvement_rate = improved_moves / max(1, accepted_moves)
                console.print(f"Temp: {temperature:.2f}, Accept rate: {acceptance_rate:.2f}, Improve rate: {improvement_rate:.2f}")
        
        # Resfriamento
        temperature *= alpha
        
        # Adaptação dinâmica baseada em taxas de aceitação
        acceptance_rate_at_temp = accepted_at_temp / moves_at_temp
        if acceptance_rate_at_temp < 0.05:
            # Se muito poucas aceitações, desacelera o resfriamento
            alpha = min(0.99, alpha * 1.05)
            console.print(f"Slowing cooling, new alpha: {alpha}")
        elif acceptance_rate_at_temp > 0.6:
            # Se muitas aceitações, acelera o resfriamento
            alpha = max(0.80, alpha * 0.95)
            console.print(f"Accelerating cooling, new alpha: {alpha}")
            
        # Reaquecimento se estagnado
        if accepted_at_temp == 0:
            stagnation_counter += 1
            if stagnation_counter >= stagnation_limit:
                # Reaquecimento
                old_temp = temperature
                temperature *= reheat_factor
                console.print(f"[bold yellow]Reheating from {old_temp:.2f} to {temperature:.2f}[/bold yellow]")
                # Muda a solução atual com mutação forte
                current = mutate(best, data, 0.4, "strong")
                current_fitness = calculate_fitness(current, data)
                stagnation_counter = 0
    
    console.print(f"[bold green]Simulated Annealing completed after {total_iterations} iterations.[/bold green]")
    console.print(f"Best fitness: {best_fitness}")
    
    return best

def tabu_search(data, max_iterations=500, tabu_size=50, neighbors_per_iter=30):
    """
    Tabu Search melhorado
    """
    # Inicializa com uma solução balanceada
    current = create_initial_solution(data, "balanced")
    current_fitness = calculate_fitness(current, data)
    best = current
    best_fitness = current_fitness
    
    console.print("[bold green]Running Tabu Search...[/bold green]")
    
    # Estruturas Tabu
    tabu_list = deque(maxlen=tabu_size)
    tabu_attributes = {}  # Armazena atributos proibidos (paciente, enfermaria, dia)
    
    # Mecanismo de aspiração: movimento tabu pode ser aceito se produzir melhor solução já encontrada
    
    # Estratégias de diversificação e intensificação
    diversification_counter = 0
    max_diversification = 50
    intensification_triggered = False
    
    # Hash simplificado para agenda
    def hash_schedule(schedule):
        return hash(frozenset((pid, ward, day) for pid, (ward, day, _) in schedule.items()))
    
    for iteration in range(max_iterations):
        # Escolhe estratégia com base na fase da busca
        if diversification_counter > max_diversification:
            # Diversificação
            console.print(f"[bold cyan]Diversification at iteration {iteration}[/bold cyan]")
            # Gera nova solução com alta mutação
            current = mutate(best, data, 0.5, "strong")
            current_fitness = calculate_fitness(current, data)
            diversification_counter = 0
            
        elif intensification_triggered and iteration % 50 == 0:
            # Intensificação
            console.print(f"[bold cyan]Intensification at iteration {iteration}[/bold cyan]")
            # Retorna à melhor solução e explora sua vizinhança
            current = best.copy()
            current_fitness = best_fitness
            intensification_triggered = False
        
        # Gera vizinhos
        neighborhood_strategy = "mixed"
        neighbors = []
        
        # Gera vizinhos não tabu ou com aspiração
        attempts = 0
        max_attempts = 100
        
        while len(neighbors) < neighbors_per_iter and attempts < max_attempts:
            attempts += 1
            neighbor_strategy = random.choice(["patient_swap", "day_shift", "ward_change"])
            candidate_neighbors = get_neighbors(current, data, 1, neighbor_strategy)
            
            if not candidate_neighbors:
                continue
                
            neighbor = candidate_neighbors[0]
            neighbor_hash = hash_schedule(neighbor)
            
            # Verifica se o vizinho está na lista tabu
            is_tabu = neighbor_hash in tabu_list
            
            # Verifica atributos tabu
            for pid, (ward, day, _) in neighbor.items():
                if (pid, ward) in tabu_attributes and tabu_attributes[(pid, ward)] > iteration:
                    is_tabu = True
                    break
                if (pid, day) in tabu_attributes and tabu_attributes[(pid, day)] > iteration:
                    is_tabu = True
                    break
            
            # Calcula fitness do vizinho
            neighbor_fitness = calculate_fitness(neighbor, data)
            
            # Aspiração: aceita movimento tabu se for melhor que o melhor global
            if not is_tabu or neighbor_fitness > best_fitness:
                neighbors.append((neighbor, neighbor_fitness, neighbor_hash))
        
        if not neighbors:
            console.print("[yellow]No non-tabu neighbors found, generating mutation[/yellow]")
            mutated = mutate(current, data, 0.3, "strong")
            mutated_fitness = calculate_fitness(mutated, data)
            mutated_hash = hash_schedule(mutated)
            neighbors = [(mutated, mutated_fitness, mutated_hash)]
        
        # Encontra o melhor vizinho
        neighbor, neighbor_fitness, neighbor_hash = max(neighbors, key=lambda x: x[1])
        
        # Atualiza a lista tabu e adiciona atributos tabu
        tabu_list.append(neighbor_hash)
        
        # Adiciona atributos modificados à lista tabu
        for pid, (ward, day, _) in neighbor.items():
            if pid in current and current[pid] != neighbor[pid]:
                old_ward, old_day, _ = current[pid]
                # Proíbe voltar à mesma enfermaria ou dia por algumas iterações
                if old_ward != ward:
                    tabu_attributes[(pid, old_ward)] = iteration + random.randint(5, 15)
                if old_day != day:
                    tabu_attributes[(pid, old_day)] = iteration + random.randint(5, 15)
        
        # Atualiza solução atual
        current = neighbor
        current_fitness = neighbor_fitness
        
        # Atualiza melhor solução
        if current_fitness > best_fitness:
            best = current.copy()
            best_fitness = current_fitness
            console.print(f"Iteration {iteration}: New best fitness = {best_fitness}")
            intensification_triggered = True
            diversification_counter = 0
        else:
            diversification_counter += 1
        
        # Imprime progresso a cada 50 iterações
        if iteration % 50 == 0:
            console.print(f"Iteration {iteration}: Current fitness = {current_fitness}, Best = {best_fitness}")
    
    console.print(f"[bold green]Tabu Search completed after {max_iterations} iterations.[/bold green]")
    console.print(f"Best fitness: {best_fitness}")
    
    return best

def print_schedule(schedule, data, algorithm_name=""):
    """
    Imprime a agenda em formato de tabela, incluindo estatísticas
    """
    allocation_grid = {
        ward: {
            "capacity": data["wards"][ward]["bed_capacity"],
            "days": {day: [] for day in range(data["days"])}
        } for ward in data["wards"]
    }
    
    # Preenche a grade de alocação
    for patient_id, (ward, admission_day, stay_duration) in schedule.items():
        patient = next(p for p in data["patients"] if p["patient_id"] == patient_id)
        for day_offset in range(stay_duration):
            current_day = admission_day + day_offset
            if current_day < data["days"]:
                allocation_grid[ward]["days"][current_day].append(patient_id)
    
    # Calcula estatísticas
    total_patients = len(schedule)
    total_bed_days = sum(
        len(allocation_grid[ward]["days"][day]) 
        for ward in allocation_grid 
        for day in range(data["days"])
    )
    max_occupancy = max(
        len(allocation_grid[ward]["days"][day]) 
        for ward in allocation_grid 
        for day in range(data["days"])
    )
    
    fitness_value = calculate_fitness(schedule, data)
    
    # Cria a tabela
    title = f"Patient Allocation Overview - {algorithm_name}" if algorithm_name else "Patient Allocation Overview"
    table = Table(title=title, show_header=True, show_lines=True)
    
    # Adiciona colunas
    table.add_column("Ward\n(Capacity)", justify="center")
    for day in range(data["days"]):
        table.add_column(f"Day {day}", justify="left", overflow="fold")
    
    # Adiciona linhas de enfermarias
    for ward in allocation_grid:
        capacity = allocation_grid[ward]["capacity"]
        row_items = [f"{ward}\n({capacity} beds)"]
        daily_totals = []
        
        for day in range(data["days"]):
            patients = allocation_grid[ward]["days"][day]
            patient_list = "\n".join(patients) if patients else "-"
            row_items.append(patient_list)
            daily_totals.append(str(len(patients)))
            
        table.add_row(*row_items)
        table.add_row("TOTAL", *daily_totals, style="bold yellow")
    
    # Adiciona totais por dia
    column_totals = ["DAY TOTAL"]
    for day in range(data["days"]):
        day_total = sum(
            len(allocation_grid[ward]["days"][day]) 
            for ward in allocation_grid
        )
        column_totals.append(str(day_total))
    
    table.add_row(*column_totals, style="bold blue")
    
    # Adiciona estatísticas gerais
    console.print("\n[bold green]Estatísticas da Alocação:[/bold green]")
    console.print(f"Total de pacientes alocados: {total_patients}")
    console.print(f"Total de dias-leito: {total_bed_days}")
    console.print(f"Ocupação máxima diária: {max_occupancy}")
    console.print(f"Valor de fitness: {-fitness_value}")  # Converte de volta para custo
    
    console.print(table)

def compare_algorithms(data, print_full_results=False):
    """
    Executa e compara os quatro algoritmos
    """
    # Define os seeds aleatórios para cada algoritmo
    algorithms = {
        "Genetic Algorithm": (genetic_algorithm, {"population_size": 50, "generations": 100, "mutation_rate": 0.1}),
        "Hill Climbing": (hill_climbing, {"max_iterations": 500, "neighbors_per_iter": 20}),
        "Simulated Annealing": (simulated_annealing, {"initial_temp": 1000.0, "final_temp": 0.1}),
        "Tabu Search": (tabu_search, {"max_iterations": 500, "tabu_size": 50})
    }
    
    results = {}
    
    for name, (func, params) in algorithms.items():
        # Define seed específico para cada algoritmo para garantir reprodutibilidade
        # mas com resultados diferentes entre algoritmos
        random.seed(data["seed"] + hash(name) % 1000)
        
        console.print(f"\n[bold magenta]Running {name}...[/bold magenta]")
        start_time = datetime.now()
        
        # Executa o algoritmo
        solution = func(data, **params)
        
        # Calcula o tempo e o fitness
        time_taken = datetime.now() - start_time
        fitness = calculate_fitness(solution, data)
        
        results[name] = {
            "solution": solution,
            "fitness": fitness,
            "time": time_taken
        }
        
        console.print(f"[bold green]{name} completed in {time_taken.total_seconds():.2f} seconds[/bold green]")
        console.print(f"Fitness (negativo do custo): {fitness}")
        
        if print_full_results:
            print_schedule(solution, data, name)
        
    # Tabela comparativa
    comparison_table = Table(title="Comparação de Algoritmos", show_header=True)
    comparison_table.add_column("Algoritmo", justify="left")
    comparison_table.add_column("Fitness", justify="right")
    comparison_table.add_column("Tempo (s)", justify="right")
    
    # Encontra o melhor resultado para destacar
    best_fitness = max(results.values(), key=lambda x: x["fitness"])["fitness"]
    
    for name, result in results.items():
        style = "bold green" if result["fitness"] == best_fitness else ""
        comparison_table.add_row(
            name,
            f"{result['fitness']:.2f}",
            f"{result['time'].total_seconds():.2f}",
            style=style
        )
    
    console.print("\n")
    console.print(comparison_table)
    
    # Retorna o melhor algoritmo e sua solução
    best_algorithm = max(results.items(), key=lambda x: x[1]["fitness"])
    return best_algorithm[0], best_algorithm[1]["solution"]

if __name__ == "__main__":
    try:
        data = parse_instance_file("../database/instances/s1m2.dat")
    except FileNotFoundError:
        console.print("[bold red]Arquivo de instância não encontrado![/bold red]")
        console.print("Por favor, forneça o caminho correto para o arquivo:")
        file_path = input("Caminho do arquivo: ").strip()
        data = parse_instance_file(file_path)
    
    console.print("[bold cyan]Escolha o algoritmo que deseja executar:[/bold cyan]")
    console.print("1 - Genetic Algorithm")
    console.print("2 - Hill Climbing")
    console.print("3 - Simulated Annealing")
    console.print("4 - Tabu Search")
    console.print("5 - Compare All Algorithms")

    choice = None
    while choice not in {"1", "2", "3", "4", "5"}:
        choice = input("Algoritmo: ").strip()

    # Definir uma nova seed para o random para cada execução
    seed = data["seed"] if "seed" in data else 42
    random.seed(seed)

    if choice == "1":
        best_schedule = genetic_algorithm(data)
        print_schedule(best_schedule, data, "Genetic Algorithm")
    elif choice == "2":
        best_schedule = hill_climbing(data)
        print_schedule(best_schedule, data, "Hill Climbing")
    elif choice == "3":
        best_schedule = simulated_annealing(data)
        print_schedule(best_schedule, data, "Simulated Annealing")
    elif choice == "4":
        best_schedule = tabu_search(data)
        print_schedule(best_schedule, data, "Tabu Search")
    elif choice == "5":
        best_algo, best_schedule = compare_algorithms(data)
        console.print(f"\n[bold green]Best Algorithm: {best_algo}[/bold green]")
        print_schedule(best_schedule, data, f"Best Solution ({best_algo})")