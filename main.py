
import pandas as pd
import paito
import ast

dados = pd.read_csv("tabela.csv")

paito = paito.Grafo(repr="lista", direcionado=False, ponderado=False)

paito.adicionarVertice("A")
paito.adicionarVertice("B")
paito.adicionarVertice("C")
paito.adicionarVertice("D")
paito.adicionarVertice("E")
paito.adicionarVertice("F")
paito.adicionarVertice("G")
# paito.adicionarVertice("H")


paito.adicionarAresta('A', 'C')
paito.adicionarAresta('A', 'D')
paito.adicionarAresta('A', 'F')
paito.adicionarAresta('B', 'E')
paito.adicionarAresta('B', 'F')
paito.adicionarAresta('C', 'E')
paito.adicionarAresta('C', 'G')
paito.adicionarAresta('C', 'D')
paito.adicionarAresta('E', 'F')


print(paito.averageGeodesicDistance())