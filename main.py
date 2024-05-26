
import pandas as pd
import paito

dados = pd.read_csv("tabela.csv")

paito = paito.Grafo(repr="lista", direcionado=False, ponderado=False)

# vertices = ["A", 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

# for vertice in vertices:
#     paito.adicionarVertice(vertice)


paito.adicionarVertice("A")
paito.adicionarVertice("B")
paito.adicionarVertice("C")
paito.adicionarVertice("D")
paito.adicionarVertice("E")
paito.adicionarVertice("F")
paito.adicionarVertice("G")
paito.adicionarVertice("H")
paito.adicionarVertice("I")
paito.adicionarVertice("J")


paito.adicionarAresta("A", "B")
paito.adicionarAresta("A", "C")
paito.adicionarAresta("B", "C")
paito.adicionarAresta("B", "D")
paito.adicionarAresta("D", "E")
paito.adicionarAresta("D", "F")
paito.adicionarAresta("D", "G")
paito.adicionarAresta("F", "E")
paito.adicionarAresta("G", "I")
paito.adicionarAresta("I", "J")
paito.adicionarAresta("J", "H")
paito.adicionarAresta("I", "H")
paito.adicionarAresta("G", "H")

# print(paito)

# print(paito.pathFinder("A"))
# print(paito.qtdShortestPaths("B", "D"))
# print(paito.shortestPathsEdge("A", "E"))
# print(paito.edgeBetweenness("B", "D"))

print(paito.communityDetection(4))

# print(paito.allEdgesBet())