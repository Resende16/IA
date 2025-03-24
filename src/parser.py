import re

def parse_file(file_path):
    data = {
        "parameters": {},
        "specialisms": {},
        "wards": {},
        "patients": []
    }
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data["parameters"] = {
        "Seed": int(lines[0].split(':')[1]),
        "M": int(lines[1].split(':')[1]),
        "Weight_overtime": float(lines[2].split(':')[1]),
        "Weight_undertime": float(lines[3].split(':')[1]),
        "Weight_delay": float(lines[4].split(':')[1]),
        "Days": int(lines[5].split(':')[1]),
        "Specialisms": int(lines[6].split(':')[1])
    }
    
    idx = 7
    for _ in range(data["parameters"]["Specialisms"]):
        parts = lines[idx].split('\t')
        name = parts[0]
        factor = float(parts[1])
        values = list(map(int, parts[2].split(';')))
        data["specialisms"][name] = {"factor": factor, "values": values}
        idx += 1
    
    num_wards = int(lines[idx].split(':')[1])
    idx += 1
    for _ in range(num_wards):
        parts = lines[idx].split('\t')
        name, capacity, _, spec, relations, rel_values = parts
        capacity = int(capacity)
        relations = relations.split(';')
        rel_values = list(map(float, rel_values.split(';')))
        data["wards"][name] = {
            "capacity": capacity,
            "specialism": spec,
            "relations": {relations[i]: rel_values[i] for i in range(len(relations))}
        }
        idx += 1
    
    num_patients = int(lines[idx].split(':')[1])
    idx += 1
    for _ in range(num_patients):
        parts = lines[idx].split('\t')
        patient_id = parts[0]
        spec, admit, start, duration, severity, factors = parts[1:]
        factors = list(map(float, factors.split(';')))
        data["patients"].append({
            "id": patient_id,
            "specialism": spec,
            "admit_day": int(admit),
            "start_day": int(start),
            "duration": int(duration),
            "severity": int(severity),
            "factors": factors
        })
        idx += 1
    
    return data

# data = parse_file("/mnt/data/s1m3.dat")
# print(data)  

