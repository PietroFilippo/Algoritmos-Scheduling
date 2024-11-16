import pulp
import random
import numpy as np
from typing import List, Dict, Tuple

class GerenciamentoTurnos:
    def __init__(self, n_funcionarios: int, n_dias: int, n_turnos: int, demanda_min: Dict[Tuple[int, int], int]):
        self.n_funcionarios = n_funcionarios
        self.n_dias = n_dias
        self.n_turnos = n_turnos
        self.demanda_min = demanda_min

    def algoritmo_milp(self):
        "Usa a programação linear inteira mista para resolver"
        # Cria o problema
        prob = pulp.LpProblem("GerenciamentoTurnos", pulp.LpMinimize)

        # Variáveis de decição
        x = pulp.LpVariable.dicts("turnos",((i, j, t) 
                                           for i in range (self.n_funcionarios)
                                           for j in range (self.n_dias)
                                           for t in range (self.n_turnos)),
                                           cat='Binary')
        # Função objetivo
        prob += pulp.lpSum(x[i,j,t] * self._get_custo(j,t)
                           for i in range (self.n_funcionarios)
                           for j in range (self.n_dias)
                           for t in range (self.n_turnos))
        
        #Restrições
        # Demanda mínimna
        for j in range(self.n_dias):
            for t in range(self.n_turnos):
                prob += pulp.lpSum(x[i,j,t] for i in range(self.n_funcionarios)) >= self.demanda_min[j,t]

        # Horas semanais maximas
        for i in range(self.n_funcionarios):
            prob += pulp.lpSum(x[i,j,t] * 8 for j in range(self.n_dias)
                               for t in range(self.n_turnos)) <= 36
            
        # Descanso entre os turnos
        for i in range(self.n_funcionarios):
            for j in range(self.n_dias-1):
                for t in range(self.n_turnos):
                    prob += x[i,j,t] + x[i,j+1,(t+1)%self.n_turnos] <= 1

        # Resolve o problema
        prob.solve()

        return self._format_solution(x)

    def algoritmo_genetica(self, pop_size=100, generations=1000):
        "Resolve usando o algoritmo genético"
        population = self._initialize_population(pop_size)
        best_solution = None
        best_fitness = float('inf')

        for gen in range(generations):
            #Avalia a população
            fitness_scores = [self._evaluate_solution(sol) for sol in population]

            # Atualiza a melhor solução
            min_fitness_idx = np.argmin(fitness_scores)
            if fitness_scores[min_fitness_idx] < best_fitness:
                best_fitness = fitness_scores[min_fitness_idx]
                best_solution = population[min_fitness_idx]

            # Seleção
            parents = self._select_parents(population, fitness_scores)

            #Crossover e Mutação
            new_population = []
            for i in range(0, pop_size, 2):
                child1, child2 = self._crossover(parents[i], parents[i+1])
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_population.extend([child1, child2])
            
            population = new_population
        
        return best_solution
    
    def _get_custo(self, dia: int, turno: int) -> float:
        "Calcula o custo base + adicional para cada turno"
        custo_base = 100 if turno != 2 else 120 # Custo adicional noturno
        multiplicador_fds = 1.2 if dia >= 5 else 1.0 # Custo adicional fim de semana
        return custo_base * multiplicador_fds
    
    def _initialize_population(self, pop_size: int) -> List[np.ndarray]:
        "Inicializa a população com soluções aleatórias válidas"
        population = []
        for _ in range(pop_size):
            sol = np.zeros((self.n_funcionarios, self.n_dias, self.n_turnos))
            for j in range(self.n_dias):
                for t in range(self.n_turnos):
                    # Aloca aleatóriamente funcionários respetando a demanda mínima
                    funcionarios = list(range(self.n_funcionarios))
                    random.shuffle(funcionarios)
                    for i in range(self.demanda_min[j,t]):
                        sol[funcionarios[i], j, t] = 1
            population.append(sol)
        return population
    
    def _evaluate_solution(self, solution: np.ndarray) -> float:
        "Avaliar a quantidade da solução considerando os custos e violações"
        custo = 0
        penalidade = 0

        #Custo dos turnos
        for j in range(self.n_dias):
            for t in range(self.n_turnos):
                alocado = np.sum(solution[:, j, t])
                custo += alocado * self._get_custo(j, t)
                if alocado < self.demanda_min[j, t]:
                    penalidade += 1000 * (self.demanda_min[j,t] - alocado)

        # Penalidade por violações de restrições
        for i in range(self.n_funcionarios):
            # Horas semanais
            horas = np.sum(solution[i]) * 8
            if horas > 36:
                penalidade += 1000 * (horas - 36)

            #Descanso entre turnos
            for j in range (self.n_dias-1):
                for t in range(self.n_turnos):
                    if solution[i, j, t] + solution[i,j+1,(t+1)%self.n_turnos] > 1:
                        penalidade += 1000

        return custo + penalidade
    
    def _select_parents(self, population: List[np.ndarray], fitness_scores: List[float]) -> List[np.ndarray]:
        "Seleicionar os pais usando torneio"
        parents = []
        for _ in range(len(population)):
            idx1, idx2 = random.sample(range(len(population)), 2)
            if fitness_scores[idx1] < fitness_scores[idx2]:
                parents.append(population[idx1].copy())
            else:
                parents.append(population[idx2].copy())
        return parents
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        "Realizar o crossover uniforme"
        mask = np.random.rand(*parent1.shape) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2
    
    def _mutate(self, solution: np.ndarray, rate: float = 0.01) -> np.ndarray:
        "Aplica a mutação com taxa especifica"
        mutation_mask = np.random.rand(*solution.shape) < rate
        solution[mutation_mask] = 1 - solution[mutation_mask]
        return solution
    
    def _format_solution(self, x: Dict) -> np.ndarray:
        "Converter a solução do pulp para a matriz numpy"
        solution = np.zeros((self.n_funcionarios, self.n_dias, self.n_turnos))
        for i in range(self.n_funcionarios):
            for j in range(self.n_dias):
                for t in range(self.n_turnos):
                    solution[i, j ,t] = x[i, j ,t].value()
        return solution
    
if __name__ == "__main__":
    #Teste básico
    n_funcionarios = 10
    n_dias = 3
    n_turnos = 2

    # Demanda minima de cada dia (dia , turno)
    demanda_min = { (j, t): 3 for j in range(n_dias) for t in range(n_turnos) }

    gerenciador = GerenciamentoTurnos(n_funcionarios, n_dias, n_turnos, demanda_min)


    # Solução usando milp
    print("Resolvendo com MILP...")
    resultado_milp = gerenciador.algoritmo_milp()

    # Solução usando ag
    print("Resolvendo com Algoritmo Genético...")
    resultado_ag = gerenciador.algoritmo_genetica()

    #Comparar soluções
    print("\nCusto MILP:", gerenciador._evaluate_solution(resultado_milp))
    print("\nCusto AG:", gerenciador._evaluate_solution(resultado_ag))
