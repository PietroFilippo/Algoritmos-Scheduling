import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algoritmos_scheduling import GerenciamentoTurnos

def testes_comparativos():
    "Executa testes comparando os dois algoritmos"

    #Testes com complexidade crescente
    test_cases = [
        {"funcionarios": 10, "dias": 3, "turnos": 2, "nome": "pequeno"},
        {"funcionarios": 20, "dias": 5, "turnos": 3, "nome": "médio"},
        {"funcionarios": 30, "dias": 7, "turnos": 3, "nome": "grande"},
        {"funcionarios": 50, "dias": 7, "turnos": 3, "nome": "muito grande"},
        {"funcionarios": 200, "dias": 6, "turnos": 3, "nome": "muito grande"},
        {"funcionarios": 80, "dias": 14, "turnos": 3, "nome": "enorme"}
    ]

    resultados = []

    for  case in test_cases:
        # Demanda minima
        demanda_min = { (j,t): max(3, case["funcionarios"]//5)
                       for j in range(case["dias"])
                       for t in range(case["turnos"])}
        
        gerenciador = GerenciamentoTurnos(case["funcionarios"], case["dias"],
                                          case["turnos"], demanda_min)
        
        # Teste com milp
        try:
            tempo_comeco = time.time()
            resultado_milp = gerenciador.algoritmo_milp()
            tempo_milp = time.time() - tempo_comeco
            custo_milp = gerenciador._evaluate_solution(resultado_milp)
            milp_viavel = True
        except:
            tempo_milp = None
            custo_milp = None
            milp_viavel = False

        # Teste algoritmo genético
        try:
            tempo_comeco = time.time()
            resultado_ag = gerenciador.algoritmo_genetica()
            tempo_ag = time.time() - tempo_comeco
            custo_ag = gerenciador._evaluate_solution(resultado_ag)
        except:
            tempo_ag = None
            custo_ag = None

        resultados.append({
            "Caso": case["nome"],
            "Funcionários": case["funcionarios"],
            "Dias": case["dias"],
            "Turnos": case["turnos"],
            "Tempo MILP": tempo_milp,
            "Custo MILP": custo_milp,
            "MILP Viável": milp_viavel,
            "Tempo AG": tempo_ag,
            "Custo AG": custo_ag
        })

    return pd.DataFrame(resultados)

def plot_resultados(resultados_df):
    "Gera visualizações dos resultados em um gráfico"

    casos = resultados_df["Caso"]

    #Comparação de custos
    plt.figure(figsize=(10, 6))
    plt.plot(casos, resultados_df["Custo MILP"], 'bo-', label='MILP')
    plt.plot(casos, resultados_df["Custo AG"], 'ro-', label='AG')
    plt.xlabel('Tamanho do Problema')
    plt.ylabel('Custo da Solução')
    plt.title('Comparação de Qualidade das Soluções')
    plt.legend()
    plt.tight_layout()

if __name__ == "__main__":
    #Executar testes
    resultados = testes_comparativos()

    # Resultados
    print("\nResultados dos testes:")
    print(resultados)

    # Gráficos
    plot_resultados(resultados)
    plt.show()