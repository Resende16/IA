from rich.console import Console
from parser import parse_instance_file
from utils import print_schedule
from genetic_algorithm import genetic_algorithm
from hill_climbing import hill_climbing
from simulated_annealing import simulated_annealing
from tabu_search import tabu_search

console = Console()

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
    elif choice == "2":
        best_schedule = hill_climbing(data)
    elif choice == "3":
        best_schedule = simulated_annealing(data)
    elif choice == "4":
        best_schedule = tabu_search(data)

    if 'best_schedule' in locals():
        print_schedule(best_schedule, data)