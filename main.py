import pandas as pd
from sklearn.metrics.cluster import v_measure_score
import random
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, Manager
from multiprocessing import Value

random.seed(42)

def get_data(dataset_name, normalize=True):
    dataset_test_path = f'./data/{dataset_name}-train.csv'
    dataset_train_path = f'./data/{dataset_name}-test.csv'
    if dataset_name == 'breast_cancer_coimbra':
        df_train = pd.read_csv(dataset_train_path)
        df_test = pd.read_csv(dataset_test_path)

        X_train = df_train.drop(columns=['Classification']).to_numpy()
        y_train = df_train['Classification'].to_numpy()
        X_test = df_test.drop(columns=['Classification']).to_numpy()
        y_test = df_test['Classification'].to_numpy()
        
    elif dataset_name == 'wineRed':
        df_train = pd.read_csv(dataset_train_path, header=None)
        df_test = pd.read_csv(dataset_test_path, header=None)

        X_train = df_train.drop(df_train.columns[-1], axis=1).to_numpy()
        y_train = df_train[df_train.columns[-1]].to_numpy()
        X_test = df_test.drop(df_test.columns[-1], axis=1).to_numpy()
        y_test = df_test[df_test.columns[-1]].to_numpy()
    
    if not normalize:
        return X_train, y_train, X_test, y_test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test 



class SafeMath:
    @staticmethod
    def divide(a, b):
        return a / 0.0000000001 if b == 0 else a / b
    @staticmethod
    def log(x):
        if x == 0:
            x += 0.0000000001
        return math.log(abs(x)) 
    @staticmethod
    def sqrt(x):
        return math.sqrt(abs(x))
    @staticmethod
    def pow2(x):
        return x * x
    @staticmethod
    def inv(x):
        if(x == 0):
            x += 0.0000000001
        return 1 / x

#dataset_name = 'wineRed'
dataset_name = 'breast_cancer_coimbra'
X_train, y_train, X_test, y_test = get_data(dataset_name)
n_features = X_train.shape[1]

RANGE_CONST = 10
NIVEIS_RECURSAO = 2
terminais = ["+", "-", "*", "SafeMath.inv", "(", ")"] \
            + [str(i) for i in range(1, RANGE_CONST)] \
            + [f"dx[{i}]" for i in range(n_features)]
naoterminais = ["<S>", "<E>", "<OP>", "<POP>", "<VAR>", "<CONST>"] + [f"<E_{i}>" for i in range(NIVEIS_RECURSAO)]
gramatica = {
    "<S>" : [["<E>"]],
    "<OP>" : [["+"], ["-"], ["*"]],
    "<E>" : [["<E_0>", "<OP>", "<E_0>"], ["<POP>", "(", "<E_0>", ")"], ["<VAR>"]],
    "<POP>" : [["SafeMath.inv"], ["<CONST>", "*"]],
    "<CONST>" : [[str(i)] for i in range(1, RANGE_CONST)],
    "<VAR>" : [[f"dx[{i}]"]  for i in range(n_features)]
}
#    
for i in range(NIVEIS_RECURSAO - 1):
    gramatica[f'<E_{i}>'] = [[f"<E_{i+1}>", "<OP>", f"<E_{i+1}>"], ["<POP>", "(", f"<E_{i+1}>", ")"], \
                             ["<VAR>"]]
gramatica[f'<E_{NIVEIS_RECURSAO-1}>'] = [["<VAR>", "<OP>", "<VAR>"], \
                                       ["<POP>", "(", "<VAR>", ")"], \
                                       ["<VAR>"]]

class Individuo():
    def __init__(self, genotipo = None):
        if(genotipo is not None):
            self.genotipo = genotipo
            self.fenotipo = self.calcular_fenotipo()
            return
        
        self.genotipo = {}
        self.genotipo["<S>"] = [0]
        self.genotipo["<E>"] = [random.randint(0, len(gramatica["<E>"]) - 1)]
        for i in range(NIVEIS_RECURSAO):
            self.genotipo[f"<E_{i}>"] = [random.randint(0, len(gramatica[f"<E_{i}>"]) - 1) for _ in range(2**(i+1))]
        self.genotipo["<OP>"] = [random.randint(0, len(gramatica["<OP>"]) - 1) for _ in range(2**(NIVEIS_RECURSAO + 1))]
        self.genotipo["<POP>"] = [random.randint(0, len(gramatica["<POP>"]) - 1) for _ in range(2**(NIVEIS_RECURSAO + 1))]
        self.genotipo["<VAR>"] = [random.randint(0, len(gramatica["<VAR>"]) - 1) for _ in range(2**(NIVEIS_RECURSAO + 1))]
        self.genotipo["<CONST>"] = [random.randint(0, len(gramatica["<CONST>"]) - 1) for _ in range(2**(NIVEIS_RECURSAO + 1))]
        
        self.fenotipo = self.calcular_fenotipo()

    def calcular_fenotipo(self):
        ids_expansao = {}
        for nao_terminal in gramatica.keys():
            ids_expansao[nao_terminal] = 0
        fronteira = ["<S>"]
        ind_no_a_expandir = 0

        while ind_no_a_expandir < len(fronteira):
            if fronteira[ind_no_a_expandir] in terminais:    
                ind_no_a_expandir += 1
                continue
            nao_terminal = fronteira[ind_no_a_expandir]
            if(ids_expansao[nao_terminal] >= len(self.genotipo[nao_terminal])):
                print(f"Erro : {nao_terminal} - {ids_expansao[nao_terminal]} - len: {len(self.genotipo[nao_terminal])}")                
            ind_producao = self.genotipo[nao_terminal][ids_expansao[nao_terminal]]
            ids_expansao[nao_terminal] += 1
            producao = gramatica[nao_terminal][ind_producao]
            fronteira.pop(ind_no_a_expandir)
            fronteira = fronteira[:ind_no_a_expandir] + producao + fronteira[ind_no_a_expandir:]
        return ''.join(fronteira)
    
    def fazer_mutacao(self):
        terminal_aleatorio = random.choice(list(gramatica.keys()))
        lista = self.genotipo[terminal_aleatorio]
        indice_aleatorio = random.randint(0, len(lista) - 1)
        producao_aleatoria = random.randint(0, len(gramatica[terminal_aleatorio]) - 1)
        self.genotipo[terminal_aleatorio][indice_aleatorio] = producao_aleatoria
        self.fenotipo = self.calcular_fenotipo()

    def calcular_distancia(self, xi, xj):
        dx = abs(xi - xj)
        return eval(self.fenotipo, {"math": math, "dx": dx, "SafeMath": SafeMath})
    

    def fitness(self, X, y):
        n_instancias = len(X)
        n_labels = len(set(y)) 
        matriz_distancias = np.zeros((n_instancias, n_instancias))
        for i in range(n_instancias):
            for j in range(i, n_instancias):
                distancia = self.calcular_distancia(X[i], X[j])
                matriz_distancias[i][j] = matriz_distancias[j][i] = distancia
        clustering = AgglomerativeClustering(
        n_clusters=n_labels,   
        metric='precomputed', 
        linkage='complete' 
        )
        labels = clustering.fit_predict(matriz_distancias)
        return v_measure_score(y, labels)

class Populacao():
    def __init__(self, n_pop, pc, pm, tam_torneio, X, y, eh_elitismo=False):
        self._populacao: list[Individuo] = [Individuo() for _ in range(2 * n_pop)]
        self._n_pop = n_pop
        self._pc = pc
        self._pm = pm
        self._tam_torneio = tam_torneio
        self._eh_elitismo = eh_elitismo
        self._X = X
        self._y = y
        self._n_processos = 10
        self._todas_fitness = Manager().list([0.0 for _ in range(2 * n_pop)])
                  
        self.gerar_pop_inicial()

    def avaliar_fitness_individuo_i(self, i):    
        fitness = self._populacao[i].fitness(self._X, self._y)
        self._todas_fitness[i] = fitness

    def gerar_pop_inicial(self):
        with ProcessPoolExecutor(max_workers=self._n_processos) as executor:
            executor.map(self.avaliar_fitness_individuo_i, range(self._n_pop))
        print(f"fitness: {self._todas_fitness}")


    def escolher_ind_pais(self):
        ind_pai1 = random.randint(0, self._n_pop - 1)
        while True:
            ind_pai2 = random.randint(0, self._n_pop - 1)
            if ind_pai1 != ind_pai2:
                break
        return ind_pai1, ind_pai2

    def cruzamento(self, pai1, pai2):
        genotipo_filho = {}
        for nao_terminal in gramatica.keys():
            if(random.random() < 0.5):
                genotipo_filho[nao_terminal] = pai1.genotipo[nao_terminal]
            else:
                genotipo_filho[nao_terminal] = pai2.genotipo[nao_terminal]
        return Individuo(genotipo_filho)

    def gerar_filho_i(self, i):
        ind_pai1, ind_pai2 = self.escolher_ind_pais()
        pai1 = self._populacao[ind_pai1]
        fit_pai1 = self._todas_fitness[ind_pai1]
        pai2 = self._populacao[ind_pai2]
        fit_pai2 = self._todas_fitness[ind_pai2]

        if random.random() < self._pc:
            filho = self.cruzamento(pai1, pai2)
            fitness_filho = filho.fitness(self._X, self._y)
            melhor_que_pais = fitness_filho > ((fit_pai1 + fit_pai2) / 2)
        else:
            filho = pai1
            fitness_filho = fit_pai1
            melhor_que_pais = False
        if random.random() < self._pm:
            filho.fazer_mutacao()

        self._populacao[i] = filho
        self._todas_fitness[i] = fitness_filho
        
        if melhor_que_pais:
            print("Filho melhor que os pais")
            self._n_filhos_melhores_que_pais += 1

    def gerar_filhos(self):
        self._n_filhos_melhores_que_pais = Manager().Value('i', 0)          
        with ProcessPoolExecutor(max_workers=self._n_processos) as executor:
            executor.map(self.gerar_filho_i, range(self._n_pop, self._n_pop * 2))

        print(f"fitness: {self._todas_fitness}")

        
    def fazer_selecao(self):
        inds_ja_escolhidos = []
        if(self._eh_elitismo):
            best_fitness = self._todas_fitness[0]
            ind_melhor_fit = 0
            for i in range(len(self._populacao)):
                if(self._todas_fitness[i] > best_fitness):
                    best_fitness = self._todas_fitness[i]
                    ind_melhor_fit = i
            inds_ja_escolhidos.append(ind_melhor_fit)
            self._populacao[0] = self._populacao[ind_melhor_fit]
            self._todas_fitness[0] = self._todas_fitness[ind_melhor_fit]
        
        inicio = 1 if self._eh_elitismo else 0
        for i in range(inicio, self._n_pop):
            inds_torneio : list[int] = []
            while(len(inds_torneio) < self._tam_torneio):
                indice_aleatorio = random.randint(0, len(self._populacao) - 1)
                if(indice_aleatorio not in inds_torneio and indice_aleatorio not in inds_ja_escolhidos):
                    inds_torneio.append(indice_aleatorio)
            ind_vencedor = inds_torneio[0]
            for indice  in inds_torneio:
                if(self._todas_fitness[indice] > self._todas_fitness[ind_vencedor]):
                    ind_vencedor = i
            inds_ja_escolhidos.append(ind_vencedor)
            self._populacao[i] = self._populacao[ind_vencedor]
            self._todas_fitness[i] = self._todas_fitness[ind_vencedor]
                

    def get_fitness_media(self):
        return sum(self._todas_fitness[i] for i in range(self._n_pop)) / self._n_pop
    
    def get_melhor_individuo(self) -> tuple[Individuo, int]:
        ind_melhor = 0
        for i in range(self._n_pop):
            if self._todas_fitness[i] > self._todas_fitness[ind_melhor]:
                ind_melhor = i
        return self._populacao[ind_melhor], self._todas_fitness[ind_melhor]
    
    def get_pior_individuo(self) -> tuple[Individuo, int]:
        ind_pior = 0
        for i in range(self._n_pop):
            if self._todas_fitness[i] < self._todas_fitness[ind_pior]:
                ind_pior = i
        return self._populacao[ind_pior], self._todas_fitness[ind_pior]
    
    def get_distribuicao_pop(self):
        distribuicao = {}
        for i in range(self._n_pop):
            fenotipo = self._populacao[i].fenotipo
            if fenotipo not in distribuicao:
                distribuicao[fenotipo] = 1
            else:
                distribuicao[fenotipo] += 1
        return distribuicao

    def get_tam_pop(self):
        return len(self._populacao)


def print_status(array, nome):
    media = np.mean(array)
    std = np.std(array)
    print(f"{nome} - Média: {media:.3f} +- {std:.3f}")



if __name__ == '__main__':
    N_GERACOES = 50
    N_POP = 50
    PC = 0.9
    PM = 0.05
    TAM_TORNEIO = 2
    EH_ELITISMO = True
    N_RUNS = 1
    arr_piores_fitness = []
    arr_melhores_fitness = []
    arr_fitness_media = []
    arr_num_individuos_diferentes = []
    pop = Populacao(N_POP, PC, PM, TAM_TORNEIO, X_train, y_train, eh_elitismo = EH_ELITISMO)
    for i in range(N_GERACOES):
        tempo_inicio = time.time()
        pop.gerar_filhos()
        print(f"Filhos melhores que os pais: {pop._n_filhos_melhores_que_pais.value}")
        pop.fazer_selecao()
        tempo_fim = time.time()
        print(f'Geração {i} - Fitness médio: {pop.get_fitness_media()}')
        print(f'Melhor indivíduo: {pop.get_melhor_individuo()[0].fenotipo} - Fitness: {pop.get_melhor_individuo()[1]}')
        print(f"tamanho pop: {pop.get_tam_pop()}")
        print(f"Tamanho indivíduos diferentes: {len(pop.get_distribuicao_pop())}")
        print(f"Tempo de execução: {tempo_fim - tempo_inicio} segundos")
        print("-"*20)
        arr_piores_fitness.append(pop.get_pior_individuo()[1])
        arr_melhores_fitness.append(pop.get_melhor_individuo()[1])
        arr_fitness_media.append(pop.get_fitness_media())
        arr_num_individuos_diferentes.append(len(pop.get_distribuicao_pop()))

    print_status(arr_piores_fitness, "Piores Fitness")
    print_status(arr_melhores_fitness, "Melhores Fitness")
    print_status(arr_fitness_media, "Fitness Média")
    print_status(arr_num_individuos_diferentes, "Num Individuos Diferentes")

    plt.plot(arr_piores_fitness, label='Piores Fitness', color='red')
    plt.plot(arr_melhores_fitness, label='Melhores Fitness', color='green')
    plt.plot(arr_fitness_media, label='Fitness Média', color='blue')

    plt.xlabel('Index')
    plt.ylabel('Fitness Value')
    plt.title('Fitness Comparison')

    plt.legend()
    plt.savefig('fitness_comp.png', format='png', dpi=300)

    plt.show()