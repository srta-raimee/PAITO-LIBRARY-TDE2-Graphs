import pandas as pd
import paito

dados = pd.read_csv("tabela.csv")

paito = paito.Grafo(repr="lista", direcionado=False, ponderado=True)

paito.adicionarVertice("A")
paito.adicionarVertice("B")
paito.adicionarVertice("D")
paito.adicionarVertice("C")
paito.adicionarVertice("E")
paito.adicionarVertice("F")
paito.adicionarVertice("G")
# paito.adicionarVertice("H")
#cu

paito.adicionarAresta('A', 'C')
paito.adicionarAresta('A', 'D')
paito.adicionarAresta('A', 'F')
paito.adicionarAresta('B', 'E')
paito.adicionarAresta('B', 'F')
paito.adicionarAresta('C', 'E')
paito.adicionarAresta('C', 'D')
paito.adicionarAresta('C', 'G')
paito.adicionarAresta('E', 'F')

# paito.adicionarAresta('A', 'B')
# paito.adicionarAresta('B', 'C')
# paito.adicionarAresta('C', 'D')
# paito.adicionarAresta('D', 'A')


# paito.adicionarAresta('A', 'B')
# paito.adicionarAresta('B', 'C')
# paito.adicionarAresta('B', 'E')
# paito.adicionarAresta('B', 'F')
# # paito.adicionarAresta('C', 'G') 
# paito.adicionarAresta('C', 'D')
# paito.adicionarAresta('D', 'C')
# # paito.adicionarAresta('D', 'H')
# paito.adicionarAresta('E', 'A')
# paito.adicionarAresta('E', 'F')
# paito.adicionarAresta('F', 'G')
# paito.adicionarAresta('G', 'F')
# paito.adicionarAresta('G', 'H')


# paito.adicionarAresta('A', 'B')
# paito.adicionarAresta('B', 'C')
# paito.adicionarAresta('C', 'A')
# paito.adicionarAr esta('C', 'D')
# paito.adicionarAresta('D', 'E')
# paito.adicionarAresta('E', 'F')
# paito.adicionarAresta('F', 'D')


# print(paito.buscaLargura("A"))
print(paito.eccentricityFinder("D"))
# print(paito.eccentricity())


