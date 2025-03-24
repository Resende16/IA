import numpy as np
import random

def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = {}
    section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        if line.endswith(':'):
            section = line[:-1]
            data[section] = []
        else:
            data[section].append(line)
    
    return data

def parse_data(data):
    specialisms = {}
    wards = {}
    patients = []
    
    for line in data.get('Specialisms', []):
        parts = line.split('\t')
        specialisms[parts[0]] = {
            'weight': float(parts[1]),
            'demand': list(map(int, parts[2].split(';')))
        }
    
    for line in data.get('Wards', []):
        parts = line.split('\t')
        wards[parts[0]] = {
            'capacity': int(parts[1]),
            'specialism': parts[3]
        }
    
    for line in data.get('Patients', []):
        parts = line.split('\t')
        patients.append({
            'id': parts[0],
            'specialism': parts[1],
            'admission': int(parts[2]),
            'discharge': int(parts[3]),
            'days': int(parts[4]),
            'urgency': int(parts[5]),
            'weights': list(map(float, parts[6].split(';')))
        })
    
    return specialisms, wards, patients

def allocate_patients(specialisms, wards, patients):
    allocation = {ward: [] for ward in wards.keys()}
    
    for patient in sorted(patients, key=lambda p: p['urgency'], reverse=True):
        possible_wards = [w for w, data in wards.items() if data['specialism'] == patient['specialism']]
        
        if possible_wards:
            chosen_ward = random.choice(possible_wards)  
            allocation[chosen_ward].append(patient['id'])
    
    return allocation

def display_menu():
    print("\n### Sistema de Alocação de Pacientes ###")
    print("1 - Visualizar Dados")
    print("2 - Realizar Alocação")
    print("3 - Sair")

def main():
    file_path = 'database/instances/s1m3.dat'  
    data = load_data(file_path)
    specialisms, wards, patients = parse_data(data)
    
    while True:
        display_menu()
        choice = input("Escolha uma opção: ")
        
        if choice == '1':
            print("Especialidades:", specialisms)
            print("Wards:", wards)
            print("Pacientes:", len(patients))
        elif choice == '2':
            allocation = allocate_patients(specialisms, wards, patients)
            print("Alocação Concluída!", allocation)
        elif choice == '3':
            print("A Sair...")
            break
        else:
            print("Opção inválida!")

if __name__ == "__main__":
    main()
