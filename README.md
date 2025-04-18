

#  Alocação de Pacientes no Hospital - Inteligência Artificial

Este projeto em Python utiliza algoritmos de Inteligência Artificial para otimizar a alocação de pacientes em hospitais, considerando critérios específicos definidos no sistema. O objetivo é melhorar a eficiência e a organização dos recursos hospitalares, criando um calendário com a forma mais otimizada de alocar os diferentes pacientes, nos diferentes serviços e especialidades, conforme as suas capacidades e workloads, e ao longo de 7 dias. É fornecida uma extensa base de dados com várias instâncias, em que cada uma tem muitos dados relevantes para o projeto. Fizemos então um parser para ler todos estes dados e podermos aplicar os algoritmos. Implementamos também cada um dos 4 algoritmos separadamente, criando um menu (no main.py) no qual é possivel fazer a seleção do algoritmo a utilizar para fazer a alocação dos pacientes no hospital. Por fim, analisamos também o fitness de cada algoritmo, para avaliar a eficiência de cada schedule (Quanto mais próximo de zero, melhor o schedule). 

## Menu

O projeto dispõe de um menu, onde é possivel escolher o algoritmo com o qual se pretende construir o calendário de alocação de pacientes. Os algoritmos utilizados e expostos no menu são:
- **1**: Genetic Algorithm
- **2**: Hill Climbing
- **3**: Simulated Annealing
- **4**: Tabu Search


##  Como Executar?

Para correr o projeto, basta executar no terminal:

```bash
python3 main.py
```

## Autores

- Rodrigo Resende - up202108750
- João Pedro - up202208936
- Tomás Ferreira - up202002749