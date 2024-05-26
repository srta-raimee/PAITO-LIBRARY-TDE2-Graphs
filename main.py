
import pandas as pd
import paito

dados = pd.read_csv("tabela.csv")

paito = paito.Grafo(repr="lista", direcionado=False, ponderado=False)

paito.adicionarVertice("A")
paito.adicionarVertice("B")
paito.adicionarVertice("C")
paito.adicionarVertice("D")
paito.adicionarVertice("E")
# paito.adicionarVertice("F")
# paito.adicionarVertice("G")
# paito.adicionarVertice("H")


paito.adicionarAresta("A", "B")
paito.adicionarAresta("A", "E")
paito.adicionarAresta("B", "C")
paito.adicionarAresta("B", "D")
paito.adicionarAresta("D", "E")
paito.adicionarAresta("C", "E")
paito.adicionarAresta("C", "D")

# print(paito)

# print(paito.pathFinder("A"))
# print(paito.qtdShortestPaths("B", "E"))
# print(paito.shortestPathsEdge("A", "E"))
# print(paito.edgeBetweenness("D", "C"))

print(paito.allEdgesBet())

# print(paito.allNodesBet())