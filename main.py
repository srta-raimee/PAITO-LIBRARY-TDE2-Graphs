import pandas as pd
import paito

dados = pd.read_csv("tabela.csv")

paito = paito.Grafo(repr="lista", direcionado=True, ponderado=True)

paito.adicionarVertice("A")
paito.adicionarVertice("B")
paito.adicionarVertice("D")
paito.adicionarVertice("C")
#paito.adicionarVertice("E")
#paito.adicionarVertice("F")
#paito.adicionarVertice("G")
#paito.adicionarVertice("H")

paito.adicionarAresta('A', 'B')
paito.adicionarAresta('B', 'C')
paito.adicionarAresta('C', 'D')

"""
paito.adicionarAresta('A', 'B')
paito.adicionarAresta('B', 'C')
paito.adicionarAresta('B', 'E')
paito.adicionarAresta('B', 'F')
paito.adicionarAresta('C', 'G') 
paito.adicionarAresta('C', 'D')
paito.adicionarAresta('D', 'C')
paito.adicionarAresta('D', 'H')
paito.adicionarAresta('E', 'A')
paito.adicionarAresta('E', 'F')
paito.adicionarAresta('F', 'G')
paito.adicionarAresta('G', 'F')
paito.adicionarAresta('G', 'H')
"""


# paito.adicionarAresta('A', 'B')
# paito.adicionarAresta('B', 'C')
# paito.adicionarAresta('C', 'A')
# paito.adicionarAr esta('C', 'D')
# paito.adicionarAresta('D', 'E')
# paito.adicionarAresta('E', 'F')
# paito.adicionarAresta('F', 'D')

# print(paito)
# print(paito.buscaLargura("A"))
# print(paito.buscaProfundidadeKosaraju())
print(paito.extractComponents())
#print(paito.componentsSCC())
# print(paito.euleriano())
# print(paito.graphEccentricity())
#print(paito.eccentricity("C"))
# print(paito.componentFinder('C'))

# print(paito.componentsSCC())