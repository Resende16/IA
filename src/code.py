import random
import math
from rich.console import Console
from rich.table import Table
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
    total_cost = 0
    daily_workload = {ward: [0.0] * data["days"] for ward in data["wards"]}
    daily_beds_used = {ward: [0] * data["days"] for ward in data["wards"]}

    for patient_id, (ward, admission_day, _) in schedule.items():
        patient = next(p for p in data["patients"] if p["patient_id"] == patient_id)
        
        if patient["specialization"] != data["wards"][ward]["major_specialization"]:
            total_cost += 1e6  
        
        for day_offset in range(patient["length_of_stay"]):
            current_day = admission_day + day_offset
            if current_day >= data["days"]:
                continue  
            
            daily_workload[ward][current_day] += patient["workload_per_day"][day_offset]
            daily_beds_used[ward][current_day] += 1
            
            if admission_day > patient["earliest_admission"]:
                total_cost += data["weights"]["delay"] * (admission_day - patient["earliest_admission"])

    for ward in data["wards"]:
        for day in range(data["days"]):
            if daily_workload[ward][day] > data["wards"][ward]["workload_capacity"]:
                total_cost += data["weights"]["overtime"] * (
                    daily_workload[ward][day] - data["wards"][ward]["workload_capacity"]
                )
            
            if daily_beds_used[ward][day] > data["wards"][ward]["bed_capacity"]:
                total_cost += 1e6

    return -total_cost  

def initialize_population(data, population_size):
    population = []
    for _ in range(population_size):
        schedule = {}
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
        population.append(schedule)
    return population

def crossover(parent1, parent2, data):
    child = {}
    for patient in data["patients"]:
        pid = patient["patient_id"]
        if random.random() < 0.5:
            child[pid] = parent1[pid]
        else:
            child[pid] = parent2[pid]
    return child

def mutate(schedule, data, mutation_rate=0.1):
    mutated = schedule.copy()
    for patient in data["patients"]:
        if random.random() < mutation_rate:
            pid = patient["patient_id"]
            valid_wards = [
                ward for ward in data["wards"] 
                if data["wards"][ward]["major_specialization"] == patient["specialization"]
            ]
            new_day = random.randint(
                patient["earliest_admission"],
                min(patient["latest_admission"], data["days"] - 1)
            )
            mutated[pid] = (random.choice(valid_wards), new_day, patient["length_of_stay"])
    return mutated

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

def print_schedule(schedule, data):
    allocation_grid = {
        ward: {
            "capacity": data["wards"][ward]["bed_capacity"],
            "days": {day: [] for day in range(data["days"])}
        } for ward in data["wards"]
    }
    
    for patient_id, (ward, admission_day, stay_duration) in schedule.items():
        for day_offset in range(stay_duration):
            current_day = admission_day + day_offset
            if current_day < data["days"]:
                allocation_grid[ward]["days"][current_day].append(patient_id)
    
    table = Table(title="Patient Allocation Overview", show_header=True, show_lines=True)
    table.add_column("Ward\n(Capacity)", justify="center")
    
    for day in range(data["days"]):
        table.add_column(f"Day {day}", justify="left", overflow="fold")
    
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
        
        table.add_row(
            "TOTAL", 
            *daily_totals,
            style="bold yellow"
        )
    
    column_totals = ["DAY TOTAL"]
    for day in range(data["days"]):
        day_total = sum(
            len(allocation_grid[ward]["days"][day]) 
            for ward in allocation_grid
        )
        column_totals.append(str(day_total))
    
    table.add_row(*column_totals, style="bold blue")
    
    console.print(table)

if __name__ == "__main__":
    data = parse_instance_file("../database/instances/s1m2.dat") 

    console.print("[bold cyan]Escolhe o algoritmo que queres executar:[/bold cyan]")
    console.print("1 - Genetic Algorithm")
    console.print("2 - Hill Climbing")
    console.print("3 - Simulated Annealing")
    console.print("4 - Tabu Search")

    choice = None
    while choice not in {"1", "2", "3", "4"}:
        choice = input("Algoritmo: ").strip()

    if choice == "1":
        best_schedule = genetic_algorithm(data)
        console.print("[green]Genetic Algorithm executado com sucesso![/green]")
    elif choice == "2":
        console.print("[yellow]Hill Climbing ainda não implementado.[/yellow]")
        # best_schedule = hill_climbing(data)
    elif choice == "3":
        console.print("[yellow]Simulated Annealing ainda não implementado.[/yellow]")
        # best_schedule = simulated_annealing(data)
    elif choice == "4":
        console.print("[yellow]Tabu Search ainda não implementado.[/yellow]")
        # best_schedule = tabu_search(data)

    # Se o algoritmo foi executado (apenas se tiver um schedule válido):
    if 'best_schedule' in locals():
        print_schedule(best_schedule, data)

