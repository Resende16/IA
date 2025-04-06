
import random
from rich.console import Console
from rich.table import Table

console = Console()

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