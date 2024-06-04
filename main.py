
import pandas as pd
import paito

dados = pd.read_csv("tabela.csv")

paito = paito.Grafo(repr="lista", direcionado=False, ponderado=True)



paito.adicionarVertice("A")
paito.adicionarVertice("B")
paito.adicionarVertice("C")
paito.adicionarVertice("D")
paito.adicionarVertice("E")
paito.adicionarVertice("F")
paito.adicionarVertice("G")

paito.adicionarAresta("A", "C")
paito.adicionarAresta("A", "B")
paito.adicionarAresta("A", "D")
paito.adicionarAresta("A", "F")
paito.adicionarAresta("B", "E")
paito.adicionarAresta("B", "F")
paito.adicionarAresta("C", "E")
paito.adicionarAresta("C", "D")
paito.adicionarAresta("C", "G")
paito.adicionarAresta("E", "F")

# print(paito.allEdgesBet())
# print(paito.geodesic())

print(paito.geodesic())
#print(paito.listaDict)