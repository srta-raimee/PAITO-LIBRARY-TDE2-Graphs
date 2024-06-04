
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
paito.adicionarVertice("H")
paito.adicionarVertice("I")
paito.adicionarVertice("J")


paito.adicionarAresta("A", "B")
paito.adicionarAresta("A", "C")
paito.adicionarAresta("B", "C")
# paito.adicionarAresta("B", "D")
paito.adicionarAresta("D", "E")
paito.adicionarAresta("D", "F")
paito.adicionarAresta("F", "E")
# paito.adicionarAresta("D", "G")
paito.adicionarAresta("G", "H")
paito.adicionarAresta("G", "I")
paito.adicionarAresta("H", "I")
paito.adicionarAresta("I", "J")
paito.adicionarAresta("H", "J")

# print(paito.communityDetection(3))
components = paito.extractComponents()
maior = 0
maiorComponent = None

for component in components:
    tam = len(component)
    if tam > maior:
        maior = tam
        maiorComponent = component

print(paito.grafoComponent(maiorComponent))

# print(paito.allEdgesBet())


# print(paito.geodesic())
#print(paito.listaDict)