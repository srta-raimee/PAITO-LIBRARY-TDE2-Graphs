
import pandas as pd
import paito

dados = pd.read_csv("tabela.csv")

paito = paito.Grafo(repr="lista", direcionado=False, ponderado=True)

# vertices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

# for vertice in vertices:
#     paito.adicionarVertice(vertice)
# TODO: PRIM FOR LISTS!!

paito.adicionarVertice("A")
paito.adicionarVertice("B")
paito.adicionarVertice("C")
paito.adicionarVertice("D")
paito.adicionarVertice("E")

paito.adicionarAresta('A','B', 2)
paito.adicionarAresta('A','C', 4)
paito.adicionarAresta('B','C', 6)
paito.adicionarAresta('C','D', 1)
paito.adicionarAresta('C','D', 1)
paito.adicionarAresta('D','E', 2)
paito.adicionarAresta('C','E', 3)


print(paito)

prim, coiso = paito.prim()
print(f"Representação prim do grafo original: {prim}")
# print(paito.allEdgesBet())

# print(paito.prim()[0])
# print(paito.prim()[10])

# print(paito.pathFinder("A"))
# print(paito.qtdShortestPaths("B", "D"))
# print(paito.shortestPathsEdge("A", "E"))
# print(paito.edgeBetweenness("B", "D"))

# print(paito.communityDetection(4))
