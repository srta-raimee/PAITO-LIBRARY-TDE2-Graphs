
import pandas as pd
import paito

dados = pd.read_csv("tabela.csv")

paito = paito.Grafo(repr="lista", direcionado=True, ponderado=True)

# vertices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

# for vertice in vertices:
#     paito.adicionarVertice(vertice)
# TODO: PRIM FOR LISTS!!

# paito.adicionarVertice("0")
# paito.adicionarVertice("1")
# paito.adicionarVertice("2")
# paito.adicionarVertice("3")
# paito.adicionarVertice("4")
# paito.adicionarVertice("5")
paito.adicionarVertice("A")
paito.adicionarVertice("B")
paito.adicionarVertice("C")
paito.adicionarVertice("D")
paito.adicionarVertice("E")
paito.adicionarVertice("F")
paito.adicionarVertice("G")
paito.adicionarVertice("H")
# paito.adicionarVertice("I")
# paito.adicionarVertice("J")



paito.adicionarAresta("A", "B")
paito.adicionarAresta("A", "C")
paito.adicionarAresta("B", "C")
paito.adicionarAresta("B", "D")
paito.adicionarAresta("C", "E")
paito.adicionarAresta("D", "H")

paito.adicionarAresta("F", "E")
paito.adicionarAresta("D", "G")
paito.adicionarAresta("G", "H")
paito.adicionarAresta("G", "I")
paito.adicionarAresta("H", "I")
paito.adicionarAresta("I", "J", 2)
paito.adicionarAresta("H", "J", 5)





# paito.adicionarAresta("0", "1", 6)
# paito.adicionarAresta("0", "3", 5)
# paito.adicionarAresta("0", "2", 1)
# paito.adicionarAresta("1", "2", 2)
# paito.adicionarAresta("1", "4", 5)
# paito.adicionarAresta("2", "4", 6)
# paito.adicionarAresta("2", "5", 4)
# paito.adicionarAresta("2", "3", 2)
# paito.adicionarAresta("3", "5", 4)
# paito.adicionarAresta("4", "5", 3)
print(paito)

print(paito.prim())
#print(paito.recuperarPeso("A", "B"))
# print(paito.prim()[0])
# print(paito.prim()[10
#                    ])
# print(paito.verificarAresta('B', 'A'))

# print(paito.pathFinder("A"))
# print(paito.qtdShortestPaths("B", "D"))
# print(paito.shortestPathsEdge("A", "E"))
# print(paito.edgeBetweenness("B", "D"))

# print(paito.communityDetection(4))

# print(paito.allEdgesBet())