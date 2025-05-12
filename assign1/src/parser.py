import re

# Esta função serve para analisar a instancia selecionada no codigo main.py e, obviamente extrair todos os dados necessários.

def parse_instance_file(file_path): # O file_path é o caminho para o arquivo que contém a instancia
    data = {
        "seed": None,
        "minor_specialisms_per_ward": None,
        "weights": {},
        "days": None,
        "specialisms": {},
        "wards": {},
        "patients": []
    }
    
    # Aqui abrimos o ficheiro para ler todas as linhas

    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    index = 0
    data["seed"] = int(re.findall(r'\d+', lines[index])[0])
    index += 1
    data["minor_specialisms_per_ward"] = int(re.findall(r'\d+', lines[index])[0])
    index += 1
    
    # Pesos
    data["weights"] = {
        "overtime": float(lines[index].split(': ')[-1]),
        "undertime": float(lines[index + 1].split(': ')[-1]),
        "delay": float(lines[index + 2].split(': ')[-1])
    }
    index += 3
    
    # Dias
    data["days"] = int(lines[index].split(': ')[-1])
    index += 1
    
    # Especialidades
    num_specialisms = int(lines[index].split(': ')[-1])
    index += 1
    for _ in range(num_specialisms):
        parts = lines[index].strip().split()
        spec_id = parts[0]
        workload_factor = float(parts[1])
        ot_time = list(map(int, parts[2].split(';')))
        data["specialisms"][spec_id] = {"workload_factor": workload_factor, "ot_time": ot_time}
        index += 1
    
    # Wards
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
    
    # Pacientes
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

