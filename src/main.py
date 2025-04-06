from rich.console import Console
from parser import parse_instance_file
from utils import print_schedule, calculate_fitness
from genetic_algorithm import genetic_algorithm
from hill_climbing import hill_climbing
from simulated_annealing import simulated_annealing
from tabu_search import tabu_search

console = Console()

# A main function importa todas as outras funções, sendo elas o parser para ler os valores e todos os algoritmos.
# Além disso, dispõe de um menu que é imprimido no terminal, onde é possivel fazer a escolha do algoritmo para fazer o scheduling.

if __name__ == "__main__":
    # Aqui escolhemos a instância que vamos analisar e fazer o schedule
    data = parse_instance_file("../database/instances/s6m3.dat") 
    # Aqui escolhemos o algoritmo que vamos usar para fazer o schedule
    console.print("[bold cyan]Escolhe o algoritmo que queres executar:[/bold cyan]")
    console.print("1 - Genetic Algorithm")
    console.print("2 - Hill Climbing")
    console.print("3 - Simulated Annealing")
    console.print("4 - Tabu Search")

    choice = None
    while choice not in {"1", "2", "3", "4"}:
        choice = input("Algoritmo: ").strip()
    # Aqui executamos o algoritmo escolhido pelo user, para os dados da instancia que estamos a analisar
    if choice == "1":
        best_schedule = genetic_algorithm(data)
    elif choice == "2":
        best_schedule = hill_climbing(data)
    elif choice == "3":
        best_schedule = simulated_annealing(data)
    elif choice == "4":
        best_schedule = tabu_search(data)

    if 'best_schedule' in locals():
        # Calculate and display the fitness value
        fitness_value = calculate_fitness(best_schedule, data)
        console.print(f"[bold green]Fitness value: {fitness_value}[/bold green]")
        print_schedule(best_schedule, data)
    