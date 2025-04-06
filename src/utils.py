
import random
from rich.console import Console
from rich.table import Table

console = Console()





# Função que vai calcular o "fitness" para o schedule de pacientes, em que quando maior o valor que vai retornar, mais eficiente e melhor o schedule

def calculate_fitness(schedule, data):
    total_cost = 0

    # Aqui utilizamos matrizes

    daily_workload = {ward: [0.0] * data["days"] for ward in data["wards"]}
    daily_beds_used = {ward: [0] * data["days"] for ward in data["wards"]}

    # Vamos percorrer todos os pacientes e respetivos dados

    for patient_id, (ward, admission_day, _) in schedule.items():
        patient = next(p for p in data["patients"] if p["patient_id"] == patient_id)
        
        if patient["specialization"] != data["wards"][ward]["major_specialization"]:
            total_cost += 1e6  

        # Aqui vamos calcular o custo diário para cada paciente
        
        for day_offset in range(patient["length_of_stay"]):
            current_day = admission_day + day_offset
            if current_day >= data["days"]:
                continue  
            
            daily_workload[ward][current_day] += patient["workload_per_day"][day_offset]

            daily_beds_used[ward][current_day] += 1 # Incrementa a cada cama ocupada por pacientes
            
            if admission_day > patient["earliest_admission"]:
                total_cost += data["weights"]["delay"] * (admission_day - patient["earliest_admission"])

    # Aqui vamos verificar a capacidade de cada ward e dia

    for ward in data["wards"]:
        for day in range(data["days"]):

            # Quanto maior o workload maior o custo
            if daily_workload[ward][day] > data["wards"][ward]["workload_capacity"]:
                total_cost += data["weights"]["overtime"] * (
                    daily_workload[ward][day] - data["wards"][ward]["workload_capacity"]
                )
            # Se a capacidade de camas for excedida maior o custo
            if daily_beds_used[ward][day] > data["wards"][ward]["bed_capacity"]:
                total_cost += 1e6

    # A função retorna um valor negativo, que quanto maior (ou mais proximo de zero) mais eficiente será o scheduling
    return -total_cost







# Aqui criamos uma função que vai gerar um schedule aleatório

def initialize_population(data, population_size):
    population = []
    for _ in range(population_size):
        schedule = {}
        # Para cada paciente vai escolher uma ward compativel e um dia de entrada no hospital
        for patient in data["patients"]:
            valid_wards = [
                ward for ward in data["wards"] 
                if data["wards"][ward]["major_specialization"] == patient["specialization"]
            ]
            if not valid_wards:
                raise ValueError(f"No valid ward for patient {patient['patient_id']}")
            # Escolhe uma ward aleatoria
            chosen_ward = random.choice(valid_wards)
            # Escolhe um dia de entrada aleatorio
            admission_day = random.randint(
                patient["earliest_admission"],
                min(patient["latest_admission"], data["days"] - 1)
            )
            schedule[patient["patient_id"]] = (chosen_ward, admission_day, patient["length_of_stay"])
        population.append(schedule)
    # Retorna um schedule aleatorio    
    return population





# Aqui criamos uma função que vai gerar um schedule aleatório com base em um schedule existente, aplicando algumas mudanças em alguns pacientes 

def mutate(schedule, data, mutation_rate=0.1):

    # Copiamos o schedule para não modificar o original
    mutated = schedule.copy()

    for patient in data["patients"]:
        if random.random() < mutation_rate:
            pid = patient["patient_id"]
            # Verificamos a ward válida para o paciente
            valid_wards = [
                ward for ward in data["wards"] 
                if data["wards"][ward]["major_specialization"] == patient["specialization"]
            ]
            # Escolhemos um novo dia de entrada no hospitral para o paciente
            new_day = random.randint(
                patient["earliest_admission"],
                min(patient["latest_admission"], data["days"] - 1)
            )

            # Atualiza o paciente no schedule
            mutated[pid] = (random.choice(valid_wards), new_day, patient["length_of_stay"])
    return mutated




# Aqui usamos algumas bibliotecas para imprimir o schedule de forma mais legível, numa tabela formatada

def print_schedule(schedule, data):

    # Esta é a estrutura que organiza tudo numa grid por ward ou dia
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
    

    # Criação da tabela
    table = Table(title="Patient Allocation Overview", show_header=True, show_lines=True)
    table.add_column("Ward\n(Capacity)", justify="center")
    
    # Adionamos as colunas para cada dia
    for day in range(data["days"]):
        table.add_column(f"Day {day}", justify="left", overflow="fold")
    
    # Adicionamos as linhas para cada ward
    for ward in allocation_grid:
        capacity = allocation_grid[ward]["capacity"]
        row_items = [f"{ward}\n({capacity} beds)"]
        daily_totals = []
        

        # Aqui começa a preencher a tabela por dia
        for day in range(data["days"]):
            patients = allocation_grid[ward]["days"][day]
            patient_list = "\n".join(patients) if patients else "-"
            row_items.append(patient_list)
            daily_totals.append(str(len(patients)))
        
        table.add_row(*row_items)
        
        # Adicionamos uma row para o total
        table.add_row(
            "TOTAL", 
            *daily_totals,
        )
    
    column_totals = ["DAY TOTAL"]
    for day in range(data["days"]):
        day_total = sum(
            len(allocation_grid[ward]["days"][day]) 
            for ward in allocation_grid
        )
        column_totals.append(str(day_total))
    
    table.add_row(*column_totals)
    
    console.print(table)