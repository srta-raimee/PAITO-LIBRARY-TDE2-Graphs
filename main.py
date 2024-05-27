
import pandas as pd
import paito

dados = pd.read_csv("tabela.csv")

paito = paito.Grafo(repr="lista", direcionado=False, ponderado=True)

# vertices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

# for vertice in vertices:
#     paito.adicionarVertice(vertice)


paito.adicionarVertice("A")
paito.adicionarVertice("B")
paito.adicionarVertice("C")
paito.adicionarVertice("D")
paito.adicionarVertice("E")

paito.adicionarAresta("A", "B")
paito.adicionarAresta("A", "C")
paito.adicionarAresta("A", "E")
paito.adicionarAresta("B", "C")
paito.adicionarAresta("B", "D")
paito.adicionarAresta("D", "E")

print(paito)



# print(paito.verificarAresta('B', 'A'))

# print(paito.pathFinder("A"))
# print(paito.qtdShortestPaths("B", "D"))
# print(paito.shortestPathsEdge("A", "E"))
# print(paito.edgeBetweenness("B", "D"))

# print(paito.communityDetection(4))

# print(paito.allEdgesBet())