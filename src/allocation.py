import pandas as pd
import re
import random
import numpy as np
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich import print

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

def initialize_population(data, population_size):
    population = []
    for _ in range(population_size):
        schedule = {}
        for patient in data["patients"]:
            assigned_day = random.randint(patient["earliest_admission"], patient["latest_admission"])
            schedule[patient["patient_id"]] = (patient["specialization"], assigned_day, patient["length_of_stay"])
        population.append(schedule)
    return population

def fitness(schedule, data):
    cost = 0
    workload = {ward: [0] * data["days"] for ward in data["wards"]}
    
    for patient_id, (ward, day, stay) in schedule.items():
        patient = next(p for p in data["patients"] if p["patient_id"] == patient_id)
        for d in range(stay):
            if day + d < data["days"]:
                workload[ward][day + d] += patient["workload_per_day"][d]
    
    for ward, loads in workload.items():
        for load in loads:
            if load > data["wards"][ward]["workload_capacity"]:
                cost += (load - data["wards"][ward]["workload_capacity"]) * data["weights"]["overtime"]
    return -cost  

#Algorithm Implemmentation ----------------------------------------------------------------------------------------------------------------------

def genetic_algorithm(data, population_size=50, generations=100, mutation_rate=0.1):
    population = initialize_population(data, population_size)
    
    for _ in range(generations):
        population = sorted(population, key=lambda s: fitness(s, data), reverse=True)
        new_population = population[:10]  
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population[:20], 2)
            child = {**dict(list(parent1.items())[:len(parent1)//2]), **dict(list(parent2.items())[len(parent2)//2:])}
            if random.random() < mutation_rate:
                patient_id = random.choice(list(child.keys()))
                patient = next(p for p in data["patients"] if p["patient_id"] == patient_id)
                child[patient_id] = (random.choice(list(data["wards"].keys())), random.randint(0, data["days"] - 1), patient["length_of_stay"])
            new_population.append(child)
        population = new_population
    
    return max(population, key=lambda s: fitness(s, data))

#End Algorithm ---------------------------------------------------------------------------------------------------------------------------------

file_path = "../database/instances/s0m0.dat"  
data = parse_instance_file(file_path)
best_schedule = genetic_algorithm(data)

table = Table(title="\n[bold blue]Patient Schedule[/bold blue]")
table.add_column("Patient ID", justify="center")
table.add_column("Ward", justify="center")
table.add_column("Admission Day", justify="center")
table.add_column("Length of Stay", justify="center")
for patient, (ward, day, stay) in best_schedule.items():
    table.add_row(patient, ward, str(day), str(stay))
console.print(table)