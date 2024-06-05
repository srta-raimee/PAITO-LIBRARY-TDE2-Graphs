
import pandas as pd
import paito
import ast

dados = pd.read_csv("tabela.csv")

<<<<<<< Updated upstream
paito = paito.Grafo(repr="lista", direcionado=False, ponderado=False)
=======
paito = paito.Grafo(repr="matriz", direcionado=True, ponderado=True)

>>>>>>> Stashed changes

paito.adicionarVertice("S")
paito.adicionarVertice("A")
paito.adicionarVertice("B")
paito.adicionarVertice("C")
paito.adicionarVertice("D")
paito.adicionarVertice("E")

paito.adicionarVertice("G")
<<<<<<< Updated upstream
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
=======
paito.adicionarVertice("H")
paito.adicionarVertice("I")
paito.adicionarVertice("J")
paito.adicionarVertice("K")


paito.adicionarAresta("S", "A")
paito.adicionarAresta("A", "B")
paito.adicionarAresta("A", "C")

paito.adicionarAresta("S", "H")


paito.adicionarAresta("B", "E")
paito.adicionarAresta("B", "D")

paito.adicionarAresta("C", "G")
paito.adicionarAresta("H", "I")
paito.adicionarAresta("H", "J")
paito.adicionarAresta("I", "K")

print(paito.buscaProfundidadeComFinal("S", "G"))
>>>>>>> Stashed changes
