from time import time
import random
import numpy as np
# ploting imports:
import matplotlib.pyplot as plt
import seaborn as sns


class Grafo:

  def __init__(self, repr=None, direcionado=False, ponderado=False, arquivoPajek=None):

    self.repr = repr
    self.direcionado = direcionado
    self.ponderado = ponderado
    self.vertices = []
    self.path_cache = {}

    # aqui é onde será verificada e criada a representação do grafo
    if arquivoPajek:
      self.carregarPajek(arquivoPajek)

    else:
      if self.repr == "matriz":
        self.criarMatrizAdjacencias()

      if self.repr == "lista":
        self.listaDict = {}

  # ======================= Geodesic measure ======================= #

  # def geodesic(self):
  #   ''' The average geodesic distance is the relation between the sum of the minimum
  #   paths of all possible pairs of nodes and the maximum number of possible edges '''
  #   somas = []
  #   for u in range(len(self.vertices)-1):
  #     for v in range(u+1, len(self.vertices)):
  #         #-1 pq não contabiliza o primeiro vertice
  #       if not self.direcionado:
  #           somas.append( len(self.shortestPathsEdge(self.vertices[u], self.vertices[v])[0]) - 1)
        
  #       else:
  #         #-1 pq não contabiliza o primeiro vertice
  #         distanceUv = self.shortestPathsEdge(self.vertices[u], self.vertices[v])
  #         distanceVu = self.shortestPathsEdge(self.vertices[v], self.vertices[v])
  #         somas.append( len(distanceUv[0]) - 1 if distanceUv else 0)
  #         somas.append( len(distanceVu[0]) - 1 if distanceVu else 0)

  #   avg = sum(somas) / (len(self.vertices) * (len(self.vertices)-1)) 
  #   return avg if self.direcionado else 2 * avg 
  
  def geodesic(self):
    ''' The average geodesic distance is the relation between the sum of the minimum
    paths of all possible pairs of nodes and the maximum number of possible edges '''
    somas = []
    for u in range(len(self.vertices)-1):
        for v in range(u+1, len(self.vertices)):
            if not self.direcionado:
                path = self.shortestPathsEdge(self.vertices[u], self.vertices[v])
                if path:  # Verifica se o caminho não é vazio
                    somas.append(len(path[0]) - 1)
            else:
                distanceUv = self.shortestPathsEdge(self.vertices[u], self.vertices[v])
                distanceVu = self.shortestPathsEdge(self.vertices[v], self.vertices[u])
                if distanceUv:  # Verifica se o caminho não é vazio
                    somas.append(len(distanceUv[0]) - 1)
                else:
                    somas.append(0)
                if distanceVu:  # Verifica se o caminho não é vazio
                    somas.append(len(distanceVu[0]) - 1)
                else:
                    somas.append(0)

    avg = sum(somas) / (len(self.vertices) * (len(self.vertices)-1)) 
    return avg if self.direcionado else 2 * avg 
  

  def grafoComponent(self, component):
    novoGrafo = Grafo(repr=self.repr, direcionado=self.direcionado, ponderado=self.ponderado) 
    for element in component:
      if element not in novoGrafo.vertices:
        novoGrafo.adicionarVertice(element)
      for vizinho in self.pegaVizinhos(element):
        # novoGrafo.adicionarVertice(vizinho) 
        novoGrafo.adicionarAresta(element, vizinho) # recuperar as conexões que existem nos componentes
    return novoGrafo
 # ======================= eccentricity measure ======================= #
     
  def eccentricity(self):
    '''the purpose here is to find what's the maximum distance between one node to all of the others
      which means we can run BFS starting from every node and return the maximum distance of each '''

    eccentricities = []
    
    if not self.direcionado and self.conexo() or self.direcionado and self.SCC():
      for vertice in self.vertices:
          nodeEccentricity = self.eccentricityFinder(vertice)
          eccentricities.append(nodeEccentricity)

    return eccentricities
  
  def eccentricityFinder(self, verticeInicial):
    distancias = {}  
    queue = []
    visitados = []
    
    if self.repr == "lista":
        queue.append((verticeInicial, 0))  

        while queue:
            verticeAtual, distancia = queue.pop(0)  
            if verticeAtual not in visitados:
                visitados.append(verticeAtual)
                distancias[verticeAtual] = distancia  

                for vizinho in sorted(self.pegaVizinhos(verticeAtual)):
                    if vizinho not in visitados:
                        queue.append((vizinho, distancia + 1)) 


    else:  # para matriz
        queue.append((verticeInicial, 0))  

        while queue:
            verticeAtual, distancia = queue.pop(0)
            indiceVerticeAtual = self.vertices.index(verticeAtual)
            if verticeAtual not in visitados:
                visitados.append(verticeAtual)
                distancias[verticeAtual] = distancia

                for indice, adjacente in enumerate(self.matrizAdjacencias[indiceVerticeAtual]):
                    if adjacente != 0 and self.vertices[indice] not in visitados:
                        queue.append((self.vertices[indice], distancia + 1))

    maiorDistancia = 0
    for vertice in distancias:
      if distancias[vertice] > maiorDistancia:
         maiorDistancia = distancias[vertice]

    return maiorDistancia

 # ======================= diameter and radius measures ======================= #

  def diameter(self): 
    ''' The diameter of a graph is the maximum eccentricity value observed amongst all nodes '''
    if self.direcionado and self.SCC() or not self.direcionado and self.conexo():
      eccentricities = self.eccentricity()
      return max(eccentricities)
    else:
        raise Exception("O diâmetro de um grafo só pode ser calculado em grafos conectados.")
  
  def radius(self): 
    ''' The radius of a graph is the minimum eccentricity value observed amongst all nodes '''
    if self.direcionado and self.SCC() or not self.direcionado and self.conexo():
      eccentricities = self.eccentricity()
      return min(eccentricities)
    else:
         raise Exception("O Raio de um grafo só pode ser calculado em grafos conectados.")
  
 # ======================= centrality measures ======================= #

 # ======================= DEGREE ======================= #

  def allDegreeCentralities(self):
    ''' returns all degree centralities in the graph. It considers that the most central node is that one with the highest number of connections.
    closer to 1 = most central '''
    degreeCentralities = {}
    for vertice in self.vertices:
      degree = self.degree(vertice)
      n = len(self.vertices) - 1
      dCentrality = round(degree / n, 4)
      degreeCentralities[vertice] = dCentrality

    return degreeCentralities

  def highestDegreeCentralities(self):
    '''Returns the highest degree centralities of the graph'''
    degreeCentralities = {}
    for vertice in self.vertices:
      degree = self.degree(vertice)
      n = len(self.vertices) - 1
      dCentrality = round(degree / n, 4)
      degreeCentralities[vertice] = dCentrality

    return max(degreeCentralities.values())

# ======================= CLOSENESS ======================= #

  def closeness(self):
    ''' The purpose of Closeness is to find how close a node is from the others. As much closer it is to 1,
    more important the node is 'cause of its potential to spread informations faster :) '''
  
    closenesses = []
    
    for vertice in self.vertices:
          nodeCloseness = self.closenessFinder(vertice)
          closenesses.append((vertice, nodeCloseness))

    return closenesses

  def closenessFinder(self, verticeInicial):
    distancias = {}  
    queue = []
    visitados = []

    if not self.direcionado:
      if self.repr == "lista":
          queue.append((verticeInicial, 0))  

          while queue:
              verticeAtual, distancia = queue.pop(0)  
              if verticeAtual not in visitados:
                  visitados.append(verticeAtual)
                  distancias[verticeAtual] = distancia  

                  for vizinho in sorted(self.pegaVizinhos(verticeAtual)):
                      if vizinho not in visitados:
                          queue.append((vizinho, distancia + 1)) 

      else:  # para matriz
          queue.append((verticeInicial, 0))  

          while queue:
              verticeAtual, distancia = queue.pop(0)
              indiceVerticeAtual = self.vertices.index(verticeAtual)
              if verticeAtual not in visitados:
                  visitados.append(verticeAtual)
                  distancias[verticeAtual] = distancia

                  for indice, adjacente in enumerate(self.matrizAdjacencias[indiceVerticeAtual]):
                      if adjacente != 0 and self.vertices[indice] not in visitados:
                          queue.append((self.vertices[indice], distancia + 1))

      somaDistancias = sum(distancias.values())
      qntVertices = len(self.vertices)
      
      if somaDistancias > 0:
          closeness = (qntVertices - 1) / somaDistancias
      else:
          closeness = 0
        

    else: # directed graphs
        grafoTransposto = self.transpor()        
        if grafoTransposto.repr == "lista":
            queue.append((verticeInicial, 0))  

            while queue:
                verticeAtual, distancia = queue.pop(0)  
                if verticeAtual not in visitados:
                    visitados.append(verticeAtual)
                    distancias[verticeAtual] = distancia  

                    for vizinho in sorted(grafoTransposto.pegaVizinhos(verticeAtual)):
                        if vizinho not in visitados:
                            queue.append((vizinho, distancia + 1)) 

        else:  # para matriz
            queue.append((verticeInicial, 0))  

            while queue:
                verticeAtual, distancia = queue.pop(0)
                indiceVerticeAtual = grafoTransposto.vertices.index(verticeAtual)
                if verticeAtual not in visitados:
                    visitados.append(verticeAtual)
                    distancias[verticeAtual] = distancia

                    for indice, adjacente in enumerate(grafoTransposto.matrizAdjacencias[indiceVerticeAtual]):
                        if adjacente != 0 and grafoTransposto.vertices[indice] not in visitados:
                            queue.append((grafoTransposto.vertices[indice], distancia + 1))

        somaDistancias = sum(distancias.values())
        qntVertices = len(grafoTransposto.vertices)
        
        if somaDistancias > 0:
            closeness = somaDistancias / (qntVertices - 1)
        else:
            closeness = 0

    return round(closeness, 4)
  
# ======================= BETWEENNESS ======================= #
  def pathFinder(self, verticeInicial):
    ''' dijkstra algorithm but to find the shortest path between each node from an initial vertex '''
    predecessores = {}
    distanciaAcumulada = {}

    for vertice in self.vertices:
        distanciaAcumulada[vertice] = float('inf')
        predecessores[vertice] = []

    distanciaAcumulada[verticeInicial] = 0.0

    q = []
    for vertice in self.vertices:
        q.append(vertice)

    while len(q) > 0:
        verticeAtual = self.min(q, distanciaAcumulada)
        if verticeAtual is None:
            break
        q.remove(verticeAtual)

        for vizinho in self.pegaVizinhos(verticeAtual):
            novaDistancia = distanciaAcumulada[verticeAtual] + 1

            if novaDistancia < distanciaAcumulada[vizinho]:
                distanciaAcumulada[vizinho] = novaDistancia
                predecessores[vizinho] = [verticeAtual]
            elif novaDistancia == distanciaAcumulada[vizinho]:
                predecessores[vizinho].append(verticeAtual)

    caminhos = {}
    for vertice in self.vertices:
        caminhos[vertice] = self.reconstruirCaminhos(verticeInicial, vertice, predecessores)

    return caminhos

  def reconstruirCaminhos(self, verticeInicial, vertice, predecessores):
      if vertice == verticeInicial: # o menor caminho até ele mesmo é o próprio node
          return [[vertice]]
      if not predecessores[vertice]: # não existe caminho até o node
          return []
      
      caminhos = []
      for predecessor in predecessores[vertice]:
          for caminho in self.reconstruirCaminhos(verticeInicial, predecessor, predecessores):
              # função recursiva para retornar cada um dos caminhos mais curtos existentes entre os nodes
              caminhos.append(caminho + [vertice])
      
      return caminhos
  
  def allNodesBet(self):
    ''' A node is considered central if it is part of most of the shortest paths of all
    possible pairs of nodes in which it is not at the beginning or end of the path '''
    centralidade = {vertice: 0 for vertice in self.vertices}
    for verticeInicial in self.vertices:
        caminhos = self.pathFinder(verticeInicial)
        for vertice in self.vertices:
            if vertice != verticeInicial:
                for caminho in caminhos[vertice]:
                    # Isso aqui não conta o nó inicial nem o final
                    for v in caminho[1:-1]:
                        centralidade[v] += 1 / len(caminhos[vertice])

    n = len(self.vertices)
    normalizacao = 1 / ((n - 1) * (n - 2))

    # Convertendo o dicionário de centralidades em uma matriz NumPy
    centralidade_array = np.array(list(centralidade.values()))

    # Aplica a normalização para deixar tudo menor que 1
    centralidade_array *= normalizacao

    # Reconstrói o dicionário de centralidades com os índices corretos
    centralidade = {vertice: centralidade_array[i] for i, vertice in enumerate(centralidade)}

    return centralidade

  def highestBet(self):
    ''' returns the nodes with highest betweenness '''
    centralidade = {vertice: 0 for vertice in self.vertices}
    for verticeInicial in self.vertices:
        caminhos = self.pathFinder(verticeInicial)
        for vertice in self.vertices:
            if vertice != verticeInicial:
                for caminho in caminhos[vertice]:
                    for v in caminho[1:-1]:  # isso aqui não conta o node inicial nem o final
                        centralidade[v] += 1 / len(caminhos[vertice])
    
    
    n = len(self.vertices)
    normalizacao = 1 / ((n - 1) * (n - 2))
    for vertice in centralidade:
        centralidade[vertice] *= normalizacao # aplica a normalização para deixar tudo menor que 1
     
 
    return [(vertice) for vertice in centralidade if centralidade[vertice] == max(centralidade.values())]
  
# ======================= EDGE BETWEENNESS ======================= #
# ARRUMAR PARA O CENARIO REAL WORLD
  def qtdShortestPaths(self, verticeInicial, verticeFinal):
    ''' Returns how many possible shortest paths an edge has '''
    allPaths = self.pathFinder(verticeInicial)
    qtdShortestPaths = {destination: len(paths) for destination, paths in allPaths.items() if destination != verticeInicial}
    return qtdShortestPaths.get(verticeFinal, 0)

  def shortestPathsEdge(self, verticeInicial, verticeFinal):
    ''' Returns all of the shortest paths an edge has '''
    allPaths = self.pathFinder(verticeInicial)
    return allPaths.get(verticeFinal, [])

  # def edgeBetweenness(self, s, t):
  #   # Calculates the betweenness of a given edge s, t
  #   edge = (s, t)
  #   betweenness = 0.0

  #   vertices = np.array(self.vertices)
  #   for u in vertices:
  #       for v in vertices:
  #           if u != v:
  #               totalPaths = self.qtdShortestPaths(u, v)
  #               if totalPaths > 0:
  #                   pathsThroughE = 0
  #                   allPaths = self.shortestPathsEdge(u, v)
  #                   for path in allPaths:
  #                       edges_in_path = list(zip(path, path[1:]))
  #                       if edge in edges_in_path:
  #                           pathsThroughE += 1
  #                   betweenness += pathsThroughE / totalPaths

  #   return betweenness
  
  def edgeBetweenness(self, s, t):
    edge = (s, t)
    betweenness = 0.0
    
    for u in self.vertices:
        for v in self.vertices:
            if u != v:
                totalPaths = self.qtdShortestPaths(u, v)
                if totalPaths > 0:
                    pathsThroughE = 0
                    allPaths = self.shortestPathsEdge(u, v)
                    for path in allPaths:
                        if edge in zip(path, path[1:]):
                            pathsThroughE += 1
                    betweenness += pathsThroughE / totalPaths
    return betweenness

  def allEdgesBet(self):
    # Returns the betweenness of all edges as a dictionary
    allEdgesBet = {}
    for s in self.vertices:
        for t in self.vertices:
            if s != t: # não precisamos de um caminho de um vértice pra ele mesmo
                edgeBet = self.edgeBetweenness(s, t)
                if edgeBet > 0:
                    if not self.direcionado: # adiciona o mesmo valor tanto para s,t quanto para t,s 
                      allEdgesBet[(s, t)] = f"{edgeBet:.3f}"
                      allEdgesBet[(t, s)] = f"{edgeBet:.3f}"
                    else:
                       allEdgesBet[(s, t)] = f"{edgeBet:.3f}"
    
    return allEdgesBet
           
# ======================= EDGE BETWEENNESS ======================= #
  def communityDetection(self, qntComunidades):
    ''' separates the graph into comunities by removing the edges with highest btwns and creates a graph for each component '''
    components = len(self.extractComponents())
    allEdgesBet = self.allEdgesBet()
    grafoComunidades = []  # Lista para armazenar os novos grafos de cada comunidade

    while components < qntComunidades and allEdgesBet:
        maiorEdgeBet = 0
        maiorEdge = None

        for edge, bet in allEdgesBet.items():
            edgeBet = float(bet)
            if edgeBet > maiorEdgeBet:
                maiorEdgeBet = edgeBet
                maiorEdge = edge
        
        if maiorEdge:
            print(maiorEdge)
            self.removerAresta(maiorEdge[0], maiorEdge[1])
            components = len(self.extractComponents())
            allEdgesBet = self.allEdgesBet()
        else:
            break

    #  Criando um novo grafo para cada comunidade detectada
    newComponents = self.extractComponents()
    for component in newComponents:  
        novoGrafo = Grafo(repr=self.repr, direcionado=self.direcionado, ponderado=self.ponderado) 
        for element in component:
            novoGrafo.adicionarVertice(element)
            for vizinho in self.pegaVizinhos(element):
               if vizinho in novoGrafo.vertices:
                novoGrafo.adicionarAresta(element, vizinho) # recuperar as conexões que existem nos componentes
        print(novoGrafo) # para visualizar o grafo
        grafoComunidades.append(novoGrafo)  # Adiciona o novo grafo à lista de grafos de comunidades

    return grafoComunidades

# ======================= manipulações básicas e auxiliares do grafo ======================= #

  def componentsSCC(self): 
    ''' "why do we need this thing???" you might be asking. It returns how many SCC we have in the graph. we'll need it, trust me.'''
    if self.direcionado:
      originalDFS = self.buscaProfundidadeKosaraju()
      grafoTransposto = self.transpor()
      # everytime it stops running, it means a new component exists
      components = grafoTransposto.componentFinder(originalDFS[0])
      
      return components
        
      
    else:
      raise Exception("Componentes fortemente conectados só podem ser verificados em grafos direcionados")

  def extractComponents(self):
    '''extracts all of the components in a graph using DFS '''
    if self.direcionado:
      return self.componentsSCC() # func that finds each component of a directed graph

    else: # for undirected graphs
      naoVisitados = self.vertices[:]
      components = []

      while naoVisitados:
        verticeAtual = random.choice(naoVisitados)
        visitados = self.buscaProfundidade(verticeAtual)
        naoVisitados = list(set(naoVisitados) - set(visitados.keys()))
        components.append(visitados.keys())

      return components

  def buscaProfundidadeKosaraju(self):
      ''' Exaustive DFS to find each component of the graph using the Kosaraju's Algorithm'''
      # vertifica os não visitados com um laço for. Se ainda existirem no final, pega um random e reinicia a busca
      stack = []
      visitados = {}
      naoVisitados = self.vertices[:]
      visited_finished = { vertice : [None, None]  for vertice in self.vertices}
      cont = 1
      first = True
      verticeInicial = random.choice(self.vertices)
      # print(verticeInicial)

      while naoVisitados:
        if first:
          first = False
          stack.append(verticeInicial)
        else:
          stack.append(random.choice(naoVisitados))
            
        while stack:
            verticeAtual = stack[-1]  
            # print(naoVisitados.remove(verticeAtual)) # VERIFICADOR
            if verticeAtual not in visitados:
                visitados[verticeAtual] = True
                naoVisitados.remove(verticeAtual)
                visited_finished[verticeAtual][0] = cont
                cont += 1

                if self.repr == "matriz":
                    indiceVerticeAtual = self.vertices.index(verticeAtual)
                    for indice, adjacente in enumerate(self.matrizAdjacencias[indiceVerticeAtual]):
                        if adjacente != 0 and self.vertices[indice] not in visitados:
                            stack.append(self.vertices[indice])

                else:  # para lista
                    for vizinho, _ in self.listaDict.get(verticeAtual, []):
                        if vizinho not in visitados:
                            stack.append(vizinho)
        

            else:
                if visited_finished[verticeAtual][1] is None:  
                    visited_finished[verticeAtual][1] = cont
                    cont += 1
                stack.pop()


      # if len(naoVisitados) > 0:
      #   stack.append(random.choice(naoVisitados))  

      # verticesFinalizados = [v for v in visited_finished.keys() if visited_finished[v][1] is not None]
      # print(visited_finished)
      verticesOrdenados = sorted(visited_finished, key=lambda x: visited_finished[x][1], reverse=True)

     

      # return [(v, visited_finished[v]) for v in verticesOrdenados]
      return verticesOrdenados

  def componentFinder(self, verticeInicial): 
    ''' it finds the components of a directed graph '''
    stack = []
    visitados = {}
    naoVisitados = self.vertices[:]
    first = True
    components = [] # this is a list of lists; contains each component of a graph
    component = []

    while naoVisitados:
      if first:
        first = False
        stack.append(verticeInicial)
      else:
        stack.append(random.choice(naoVisitados))
        
          
      while stack:
          verticeAtual = stack[-1]  
          if verticeAtual not in visitados:
              visitados[verticeAtual] = True
              component.append(verticeAtual)
              naoVisitados.remove(verticeAtual)
              

              if self.repr == "matriz":
                  indiceVerticeAtual = self.vertices.index(verticeAtual)
                  for indice, adjacente in enumerate(self.matrizAdjacencias[indiceVerticeAtual]):
                      if adjacente != 0 and self.vertices[indice] not in visitados:
                          stack.append(self.vertices[indice])

              else:  # para lista
                  for vizinho, _ in self.listaDict.get(verticeAtual, []):
                      if vizinho not in visitados:
                          stack.append(vizinho)
      

          else:
              stack.pop()

      components.append(component)
      component = []

    # uncomment if you wanna see the begin/finish times
    # verticesFinalizados = [v for v in visited_finished.keys() if visited_finished[v][1] is not None]
    # print(visited_finished)
    # verticesOrdenados = sorted(visited_finished, key=lambda x: visited_finished[x][1], reverse=True)

    

    # return [(v, visited_finished[v]) for v in verticesOrdenados]
    return components

  def adicionarVertice(self, vertice):
    ''' adds nodes to our graph, both directed and undirected '''
    if vertice not in self.vertices:
      self.vertices.append(vertice)

      # Se a representacao for lista:
      if self.repr == "lista":
        self.listaDict[vertice] = []

      # Se for matriz:
      else:
        n = len(self.matrizAdjacencias)
        self.matrizAdjacencias.append([0] * n)  # Adiciona uma linha nova
        for linha in self.matrizAdjacencias:  # Adiciona um 0 a mais em todas as linhas
          linha.append(0)

    else:
      print(f"A vertice {vertice} já existe")

  def removerVertice(self, vertice):
    ''' removes nodes from our graph, both directed and undirected '''
    if vertice in self.vertices:

      if self.repr == "lista":

        # cria um dict novo que exclui o vértice removido
        for v in self.listaDict:
          self.listaDict[v] = [(chave, valor)
                               for chave, valor in self.listaDict[v]
                               if chave != vertice]

        # Remove todas as arestas que tem o vertice como origem
        del self.listaDict[vertice]

      else:  # caso seja matriz
        indiceVertice = self.vertices.index(vertice)
        del self.vertices[indiceVertice]

        # Remove a linha correspondente ao vértice
        del self.matrizAdjacencias[indiceVertice]

        # Remove a coluna correspondente ao vértice
        for linha in self.matrizAdjacencias:
          del linha[indiceVertice]

    else:
      print(f"Vértice '{vertice}' não encontrado no grafo.")

  def removerAresta(self, vertice1, vertice2):
    ''' removes edges from our graph, both directed and undirected '''
    if self.verificarVertice(vertice1, vertice2):

      if self.repr == "matriz":
        indiceVertice1 = self.vertices.index(vertice1)
        indiceVertice2 = self.vertices.index(vertice2)
        self.matrizAdjacencias[indiceVertice1][indiceVertice2] = 0
        if not self.direcionado:
          self.matrizAdjacencias[indiceVertice2][indiceVertice1] = 0

      else:  # self.repr == "lista":
        for vertice in self.listaDict[vertice1]:
          if vertice[0] == vertice2:
            self.listaDict[vertice1].remove((vertice))

            if not self.direcionado:
              for vertice in self.listaDict[vertice2]:
                if vertice[0] == vertice1:
                  self.listaDict[vertice2].remove((vertice))

    else:
      print("Pelo menos um dos vértices não existe no grafo.")

  def verificarVertice(self, *vertices):
    return all(vertice in self.vertices for vertice in vertices)

  def verificarAresta(self, vertice1, vertice2):
    if self.verificarVertice(vertice1, vertice2):
      if self.repr == "matriz":
        indiceVertice1 = self.vertices.index(vertice1)
        indiceVertice2 = self.vertices.index(vertice2)

        if self.direcionado and self.matrizAdjacencias[indiceVertice1][
            indiceVertice2]:
          return True

        elif not self.direcionado and self.matrizAdjacencias[indiceVertice1][
            indiceVertice2] and self.matrizAdjacencias[indiceVertice2][
                indiceVertice1]:
          return True
        return False

      else:  # para lista de adjacencias
        for vertice in self.listaDict[vertice1]:
            if vertice[0] == vertice2:
              return True
        return False
      
  def atualizarPesoAresta(self, vertice1, vertice2, novoPeso=1):
    ''' verifies if the edge exists and if the graph is weighted, so the function can update it '''

    verificar = self.verificarAresta(vertice1, vertice2)
    if verificar:  # se a aresta existe, atualiza o peso
      if self.repr == "lista":
        for vertice in self.listaDict[vertice1]:
          if vertice[0] == vertice2:
            vertice[1] = novoPeso

            # adiciona o peso na aresta inversa se não for direcionado
            if not self.direcionado:
               for vertice in self.listaDict[vertice2]:
                if vertice[0] == vertice1:
                  vertice[1] = novoPeso

      elif self.repr == "matriz":
        indiceVertice1 = self.vertices.index(vertice1)
        indiceVertice2 = self.vertices.index(vertice2)

        if self.ponderado:
          if not self.direcionado:
            self.matrizAdjacencias[indiceVertice2][indiceVertice1] = novoPeso
            self.matrizAdjacencias[indiceVertice1][indiceVertice2] = novoPeso

          else:
            self.matrizAdjacencias[indiceVertice1][indiceVertice2] = novoPeso

    else:
      if self.ponderado:
        self.adicionarAresta(vertice1, vertice2, novoPeso)

      else:
        self.adicionarAresta(vertice1, vertice2)

  def adicionarAresta(self, vertice1, vertice2, peso=1):
    ''' adds edges to our graph, both directed and undirected '''
    # Se o grafo não for ponderado, os pesos sao 1 (mesmo se passar um valor).
    if not self.ponderado:
      peso = 1

    if peso <= 0:
      print("Não é possível adicionar uma aresta com peso 0 ou menos")
      return
    
    if vertice1 not in self.vertices:
      self.adicionarVertice(vertice1)
    if vertice2 not in self.vertices:
      self.adicionarVertice(vertice2)
        
      # Para matriz de adjacencias:
    if self.repr == "matriz":
        indiceVertice1 = self.vertices.index(vertice1)
        indiceVertice2 = self.vertices.index(vertice2)
        
        self.matrizAdjacencias[indiceVertice1][indiceVertice2] = peso

        # Se o grafo não for direcionado, adicione a aresta inversa
        if not self.direcionado:
            self.matrizAdjacencias[indiceVertice2][indiceVertice1] = peso

    # para lista de adjacências
    else:
    # Percorre a lista de adjacencias do vertice1
        if self.direcionado:
          if self.verificarAresta(vertice1, vertice2):
                  return 
          else:
             self.listaDict[vertice1].append([vertice2, peso])

        # não direcionado
        else:
            
            if self.verificarAresta(vertice1, vertice2) or self.verificarAresta(vertice2, vertice1):
                  return False
            else:
             
              self.listaDict[vertice1].append([vertice2, peso])
              self.listaDict[vertice2].append([vertice1, peso])

  def pegaVizinhos(self, vertice1):
    ''' returns all the neighboors of a node, both directed and undirected '''
    if self.repr == "matriz":
      vizinhos = []

      indiceV1 = self.vertices.index(vertice1)
      for vertice in self.vertices:
        indiceV2 = self.vertices.index(vertice)

        if self.matrizAdjacencias[indiceV1][indiceV2] != 0:
          vizinhos.append(vertice)

      return vizinhos

    else: # lista
      if vertice1 in self.vertices:
            vizinhos = [vizinho for (vizinho, _) in self.listaDict[vertice1]]
            return vizinhos
      # else:
      #       return []

  def recuperarPeso(self, vertice1, vertice2):
    ''' retrives the weight between 2 nodes '''
    if self.ponderado and self.verificarAresta(vertice1, vertice2):
      if self.repr == "matriz":
        indiceVertice1 = self.vertices.index(vertice1)
        indiceVertice2 = self.vertices.index(vertice2)

        return self.matrizAdjacencias[indiceVertice1][indiceVertice2]

      else:  # lista
        for vertice in self.listaDict[vertice1]:
          if vertice[0] == vertice2:
            return vertice[1]

  # ================== Funcoes de graus ================== #
  # CONTINUAR COMENTANDO DAQUI
  # ----------------------------------------------------------------------- #
  # Indegree: Calcula quantas arestas entram no vertice, ou seja, percorre
  # todos os vertices (que não sejam o que está sendo verificado) e conta
  # quantas vezes ele aparece.
  def indegree(self, vertice):

    # --------- Direcionado
    if self.direcionado:
      # Sempre bom verificar se exite o vertice
      if vertice not in self.vertices:
        print(
            f"O vertice {vertice} não existe no grafo. Não foi possível calcular indegree"
        )
        return 0

      soma = 0

      if self.repr == "lista":
        for v in self.listaDict:  # Percorre todos os vertices
          if vertice != v:  # Pula se for a lista do vertice que está sendo verificado
            for vizin in self.listaDict[v]:
              # Se encontrar o vertice nas outras listas, soma 1:
              if vizin[0] == vertice:
                soma += 1

      else:  # self.repr == "matriz":
        i = self.vertices.index(
            vertice)  # Pega o index do vertice na lista de vertices
        for j in range(len(self.matrizAdjacencias)):
          # Percorre todas as linhas da matriz (menos a do vertice, representado pelo
          #  "j != i") e se encontrar uma ligação com o index do nosso vertice, soma 1:
          if (j != i) and (self.matrizAdjacencias[j][i] != 0):
            soma += 1

      return soma

    # --------- Não Direcionado
    else:
      # Independente do tipo de representação, se o grafo não for direcionado,
      # o indegree, outdegree e degree são a mesma coisa.
      return self.degree(vertice)

  # ----------------------------------------------------------------------- #
  # Outdegree: Calcula quantas arestas saem do vertice.
  def outdegree(self, vertice):
    # Sempre bom verificar se exite o vertice
    if vertice not in self.vertices:
      print(
          f"O vertice {vertice} não existe no grafo. Não foi possível calcular outdegree"
      )
      return 0

    soma = 0
    # --------- Direcionada
    if self.direcionado:
      if self.repr == "lista":
        return len(self.listaDict[vertice])

      else:  #self.repr == 'matriz':
        i = self.vertices.index(vertice)
        # Percorre todas as possiveis arestas de saida do vertice e conta quantas tem:
        for j in range(len(self.matrizAdjacencias[i])):
          if self.matrizAdjacencias[i][j] != 0:
            soma += 1

    # --------- Não Direcionada
    else:
      # Independente do tipo de representação, se o grafo não for direcionado,
      # o indegree, outdegree e degree são a mesma coisa.
      return self.degree(vertice)

    return soma

  # ----------------------------------------------------------------------- #
  def degree(self, vertice):
    # Sempre bom verificar se exite o vertice
    if vertice not in self.vertices:
      print(
          f"O vertice {vertice} não existe no grafo. Não foi possivel retornar o degree."
      )
      return 0

    # --------- Direcionada
    if self.direcionado:
      # Sendo um grafo direcionado, o degree é sempre a soma do grau de entrada
      # e de saída do vertice. Ou seja:
      return self.indegree(vertice) + self.outdegree(vertice)

    # --------- Não Direcionada
    else:
      if self.repr == "lista":
        return len(self.listaDict[vertice])

      else:  #self.repr == 'matriz':
        soma = 0
        i = self.vertices.index(vertice)
        for ver in range(len(self.matrizAdjacencias[i])):
          if self.matrizAdjacencias[i][ver] != 0:
            soma += 1

        return soma

  # ======================= algoritmos de busca ======================= #

  def buscaLargura(self, verticeInicial):
    inicio = time()
    if self.repr == "lista":
      visitas = {}
      queue = []
      visitados = []
      queue.append(verticeInicial)

      while queue:
        
        verticeAtual = queue.pop(0)

        if verticeAtual not in visitados:
          visitados.append(verticeAtual)

        for vizinho in sorted(self.pegaVizinhos(verticeAtual)):
          if vizinho not in visitados:
            queue.append(vizinho)

        # if verticeAtual == verticeFinal:
        #   break

        fim = time()
        tempo = fim - inicio
        visitas[verticeAtual] = (f"{tempo:.7f}")
        

    else:  # para matriz
      inicio = time()
      queue = []
      visitados = []
      visitas = {}
      queue.append(verticeInicial)

      # indiceVerticeInicial = self.vertices.index(verticeInicial)

      while queue:
        
        verticeAtual = queue.pop(0)
        indiceVerticeAtual = self.vertices.index(verticeAtual)

        if verticeAtual not in visitados:
          visitados.append(verticeAtual)

          for indice, adjacente in enumerate(
              self.matrizAdjacencias[indiceVerticeAtual]):

            if adjacente != 0 and self.vertices[indice] not in visitados:
              queue.append(self.vertices[indice])

        # if verticeAtual == verticeFinal:
        #   break

        fim = time()
        tempo = fim - inicio
        visitas[verticeAtual] = (f"{tempo:.7f}")
    return visitas

  def buscaProfundidade(self, verticeInicial):
    inicio = time()
    visitas = {}
    stack = []
    visitados = {}
    stack.append(verticeInicial)

    while stack:
      
      verticeAtual = stack.pop()

      if verticeAtual not in visitados:
        visitados[verticeAtual] = True

        if self.repr == "matriz":
          indiceVerticeAtual = self.vertices.index(verticeAtual)
          for indice, adjacente in enumerate(
              self.matrizAdjacencias[indiceVerticeAtual]):

            if adjacente != 0 and self.vertices[indice] not in visitados:
              stack.append(self.vertices[indice])

        else:  # para lista
          for vizinho, _ in self.listaDict.get(verticeAtual, []):
            if vizinho not in visitados:
              stack.append(vizinho)
              
      # if verticeAtual == verticeFinal:
      #   break
      fim = time()
      tempo = fim - inicio
      visitas[verticeAtual] = (f"{tempo:.7f}")
      
    return visitas

  def transpor(self): # transpor o grafo direcionado
      if self.direcionado:
        if self.repr == "matriz":
            transposta = Grafo("matriz", direcionado=self.direcionado, ponderado=self.ponderado)
            transposta.vertices = self.vertices
            transposta.matrizAdjacencias = [
                [self.matrizAdjacencias[j][i] for j in range(len(self.matrizAdjacencias))] for i in
                range(len(self.matrizAdjacencias))]
            return transposta
        else: # para lista
            transposta = Grafo("lista", direcionado=self.direcionado, ponderado=self.ponderado)
            transposta.vertices = self.vertices
            for vertice in self.vertices:
                transposta.listaDict[vertice] = []
            for vertice in self.vertices:
                for vizinho, peso in self.listaDict.get(vertice, []):
                    transposta.listaDict[vizinho].append((vertice, peso))
            return transposta
      else:
        return self

  def buscaDijkstra(self, verticeInicial, verticeFinal):
      if self.ponderado:
        inicio = time()
        predecessores = {}
        distanciaAcumulada = {}


        for vertice in self.vertices:
          distanciaAcumulada[vertice] = +1e10
          predecessores[vertice] = None
    
        distanciaAcumulada[verticeInicial] = 0.0
    
        q = []
        for vertice in self.vertices:
          q.append(vertice)
    
        while len(q) > 0:
          verticeAtual = self.min(q, distanciaAcumulada)
          if verticeAtual is None:
            break
          q.remove(verticeAtual)
    
          for vizinho in self.pegaVizinhos(verticeAtual):
            novaDistancia = distanciaAcumulada[verticeAtual] + self.recuperarPeso(verticeAtual, vizinho)
    
            if novaDistancia < distanciaAcumulada[vizinho]:
    
              distanciaAcumulada[vizinho] = novaDistancia
    
              predecessores[vizinho] = verticeAtual

    
        caminho = []
        distanciaTotal = 10e9
    
        if predecessores[verticeFinal] != None:
    
          distanciaTotal = distanciaAcumulada[verticeFinal]
    
          verticeAtual = verticeFinal
          while verticeAtual != None:
            caminho.insert(0, verticeAtual)
            verticeAtual = predecessores[verticeAtual]
    
        fim = time()
        tempo = fim - inicio
        return caminho, distanciaTotal, f"{tempo:.7f}"
        
      else:
        return None
      
  def min(self, q, pesosAcumulados):
      # q é uma cópia da lista de vertices
      menorCusto = None
      pesoMinimo = +1e10
      for vertice in q:
        if pesosAcumulados[vertice] <= pesoMinimo:
          pesoMinimo = pesosAcumulados[vertice]
          menorCusto = vertice
      return menorCusto

  # ======================= Persistencia (arquivo pajek) ======================= #

  def salvarPajek(self, arquivoPajek):
      
      with open(arquivoPajek, "w") as file:
        # ---- Armazenamento dos Dados:
        file.write(f"% representation={self.repr}\n")
        file.write(f"% directed={self.direcionado}\n")
        file.write(f"% weighted={self.ponderado}\n")

        # ---- Armazenamento de Vertices:
        file.write(f"*Vertices {len(self.vertices)}\n")

        for i in range(len(self.vertices)):
          file.write(f"{i} {self.vertices[i]}\n")

        # ---- Armazenamento de Arestas:
        if self.repr == "matriz":
          file.write("*arcs\n")
          # Pra cada vertice
          for i in range(len(self.matrizAdjacencias)):
            for j in range(len(self.matrizAdjacencias[i])):

              # Verifica se existe a aresta entre os vertices 'i' e 'j':
              if self.matrizAdjacencias[i][j] != 0:
                # Escreve o index do vertice de origem, de destino e por ultimo peso (se tiver)
                aresta = f"{i} {j}"

                if self.ponderado:
                  aresta += f" {self.matrizAdjacencias[i][j]}"

                file.write(f"{aresta}\n")

        else:  #self.repr == "lista":
          file.write("*edge\n")
          # Pra cada vertice de origem
          for vertice in self.listaDict:
            # Pra cada vertice ligado ao de origem
            for arestas in self.listaDict[vertice]:
              # Escreve o index do vertice de origem, de destino e por ultimo peso (se tiver)
              aresta = f"{vertice} {arestas[0]}"

              if self.ponderado:
                aresta += f" {arestas[1]}"

              file.write(f"{aresta}\n")

    # So pra deixar o carregarPajek mais limpo
  
  def clean(self, texto, retirar):
      return texto.replace(retirar, "").replace("\n", "")
    
  def carregarPajek(self, arquivoPajek):
      with open(arquivoPajek, "r") as file:
        #  ---- Dados do Grafo:
        representacao = file.readline()
        direcionamento = file.readline()
        ponderacao = file.readline()

        self.repr = self.clean(representacao, "% representation=")
        self.direcionado = bool(self.clean(direcionamento, "% directed="))
        self.ponderado = bool(self.clean(ponderacao, "% weighted="))
  
  def salvarPajek(self, arquivoPajek):
    with open(arquivoPajek, "w") as file:
      # ---- Armazenamento dos Dados:
      file.write(f"% representation={self.repr}\n")
      file.write(f"% directed={self.direcionado}\n")
      file.write(f"% weighted={self.ponderado}\n")

      # ---- Armazenamento de Vertices:
      file.write(f"*Vertices {len(self.vertices)}\n")

      for i in range(len(self.vertices)):
        file.write(f"{i} {self.vertices[i]}\n")

      # ---- Armazenamento de Arestas:
      if self.repr == "matriz":
        file.write("*arcs\n")
        # Pra cada vertice
        for i in range(len(self.matrizAdjacencias)):
          for j in range(len(self.matrizAdjacencias[i])):

            # Verifica se existe a aresta entre os vertices 'i' e 'j':
            if self.matrizAdjacencias[i][j] != 0:
              # Escreve o index do vertice de origem, de destino e por ultimo peso (se tiver)
              aresta = f"{i} {j}"

              if self.ponderado:
                aresta += f" {self.matrizAdjacencias[i][j]}"

              file.write(f"{aresta}\n")

      else:  #self.repr == "lista":
        file.write("*edge\n")
        # Pra cada vertice de origem
        for vertice in self.listaDict:
          # Pra cada vertice ligado ao de origem
          for arestas in self.listaDict[vertice]:
            # Escreve o index do vertice de origem, de destino e por ultimo peso (se tiver)
            aresta = f"{vertice} {arestas[0]}"

            if self.ponderado:
              aresta += f" {arestas[1]}"

            file.write(f"{aresta}\n")

  def clean(self, texto, retirar):
     # So pra deixar o carregarPajek mais limpo
    return texto.replace(retirar, "").replace("\n", "")

  def carregarPajek(self, arquivoPajek):
    with open(arquivoPajek, "r") as file:
      #  ---- Dados do Grafo:
      representacao = file.readline()
      direcionamento = file.readline()
      ponderacao = file.readline()

      self.repr = self.clean(representacao, "% representation=")
      self.direcionado = bool(self.clean(direcionamento, "% directed="))
      self.ponderado = bool(self.clean(ponderacao, "% weighted="))

      if self.repr == "matriz":
        self.criarMatrizAdjacencias()

      else:  # self.repr == "lista":
        self.listaDict = {}

      #  ---- Vertices
      # No arquivo pajek, a lista de vertice esta salva como:
      # *Vertices n
      # Entao criamos um 'for i' que percorra esse "n"
      n = int(self.clean(file.readline(), "*Vertices "))
      for i in range(n):
        vertice = file.readline().replace("\n", "").split(" ")
        self.adicionarVertice(vertice[1])

      #  ---- Arestas
      file.readline()  # Retira o *arcs / *edge

      if self.repr == "matriz":

        linha = file.readline()
        while linha != "":
          aresta = linha.replace("\n", "").split(" ")
          ver1 = self.vertices[int(aresta[0])]
          ver2 = self.vertices[int(aresta[1])]

          # Se a linha tiver peso, adiciona o peso:
          if self.ponderado:
            peso = int(aresta[2])
            self.adicionarAresta(ver1, ver2, peso)

          # Se nao, adiciona a aresta com 1 de "peso":
          else:
            self.adicionarAresta(ver1, ver2)

          linha = file.readline()

      else:  # self.repr == "lista":

        aresta = []
        linha = file.readline()
        while linha != "":
          aresta = file.readline().replace("\n", "").split(" ")

          # Se a linha tiver peso (3° parametro), adiciona o peso:
          if self.ponderado:
            self.listaDict[aresta[0]].append((aresta[1], int(aresta[2])))

          else:
            self.listaDict[aresta[0]].append(aresta[1], 1)

          linha = file.readline()
       

    # ======================= Funções de representação ======================= #

    # ===== cria matriz de adjacências =====

 # ======================= Funções de representação ======================= #

  # ===== cria matriz de adjacências =====
  def criarMatrizAdjacencias(self):
      n = len(self.vertices)
      self.matrizAdjacencias = [[0] * n
                                for _ in range(n)]  # Inicializa com zeros

      # Preenche com 1 onde há arestas
      for i in range(n):
        for j in range(n):
          if self.repr == "matriz" and self.matrizAdjacencias[i][j] != 0:
            self.matrizAdjacencias[i][j] = 1

    # ======================= Funções de fechamento transitivo ======================= #

  def constroiMatriz(self, qtdVertices):
      return [[0] * qtdVertices for _ in range(qtdVertices)]

  def copiaMatriz(self):
      if self.repr == "matriz":
        matriz = self.matrizAdjacencias
        qtdVertices = len(self.matrizAdjacencias)
    
        copia = self.constroiMatriz(qtdVertices)
    
        for i in range(qtdVertices):
          for j in range(qtdVertices):
            copia[i][j] = matriz[i][j]
        return copia
        
      else:
        return None
    
  def warshall(self):
      if self.repr == "matriz":
        matrizWarshall = self.copiaMatriz()
        for k in range(len(matrizWarshall)):
          for i in range(len(matrizWarshall)):
            for j in range(len(matrizWarshall)):
              matrizWarshall[i][j] = matrizWarshall[i][j] or \
              (matrizWarshall[i][k] and \
              matrizWarshall[k][j])
    
        return matrizWarshall
        
      else:
        return None
    # ======================= Função de definição de grafo euleriano ======================= #
  
  def SCC(self): # strongly connected components. Only works for directed graphs
    if self.direcionado:
      return len(self.componentsSCC()) == 1

  def euleriano(self): 
      vertice = random.choice(self.vertices)
      if self.direcionado:  # strongly connected e ter degree e outdegree iguais
        return self.SCC() and all(self.indegree(vertice) == self.outdegree(vertice) for vertice in self.vertices)
      
      else:  # não direcionado, conectado  e grau par em todos os vertices
        return all(self.degree(vertice) % 2 == 0 for vertice in self.vertices) and len(self.buscaProfundidade(vertice)) == len(self.vertices)

    # ======================= Funções MST - Árvore Geradora Mínima com algorítmo de Prim ======================= #
    
  def conexo(self): # -> grafos não direcionados
      if not self.direcionado:
        vertice = random.choice(self.vertices)
        tamanho = len(self.buscaProfundidade(vertice))
        return tamanho == len(self.vertices)
      else:
        raise Exception("A função conexo() funciona apenas para grafos não direcionados")
        
  def prim(self):
    if self.conexo() and self.ponderado:
      
        # lista de vertices e antecessores
        predecessores = {}
        pesos = {}
        for vertice in self.vertices:
            predecessores[vertice] = None
            pesos[vertice] = 1e10

        # criando lista de vertices que existem no grafo original
        q = self.vertices[:].copy()
    
        while len(q) > 0:
            # encontrar o vértice ainda não adicionado
            # que tenha o menor peso
            u = self.min(q, pesos)

            # remover esse vertice da lista
            q.remove(u)

            for vizinho in self.pegaVizinhos(u):
              peso = self.recuperarPeso(u, vizinho)
              if vizinho in q and peso < pesos[vizinho]:
                  predecessores[vizinho] = u
                  pesos[vizinho] = peso

        # monta novo grafo com as conexoes e pesos encontrados
        mst = Grafo(repr=self.repr,
                      direcionado=False,
                      ponderado=True)
        # copiar vertices originais
        for vertice in self.vertices:
          mst.adicionarVertice(vertice)

        # adiciona as arestas
        custoAcumulado = 0
        for verticeInicial in predecessores.keys():
            verticeFinal = predecessores[verticeInicial]
            if verticeFinal is not None:
              mst.adicionarAresta(verticeInicial,
                                  verticeFinal,
                                  pesos[verticeInicial])
              custoAcumulado += pesos[verticeInicial]

        #retorna a MST
        return mst, custoAcumulado
    
    else:
      return None

          
    # ===== Plotar Grafo ==== #
    
  def gerarHistograma(self, arq='histograma.png'):
    # Para pegar os dados de grau para criar o histograma:
    graus = [self.degree(x) for x in self.vertices]
    
    # Definindo a paleta de cores personalizada
    colors = ["#ffb3d9", "#ff80bf", "#ff4da6", "#ff1a8c", "#e60073", "#b30059", "#800040", "#4d0026"]

    # isso aqui apenas a paleta de cores rosa
    df = pd.DataFrame({'graus': graus})
    sns.countplot(data=df, x='graus', palette=colors, hue='graus', dodge=False, legend=False)

    # Salvar o gráfico em png
    plt.savefig(arq, format='png')
    plt.show()
    plt.close()  # Fechar o plot para evitar sobreposição em plots futuros

# ===== Printar o grafo ===== #
  
  def __str__(self):

      # Printagem dos vertices:
      toString = "\n=== Representação toString do grafo: ===\n"

      # Dados do Grafo:
      toString += "Dados do Grafo: \n"
      toString += f"  - Representação: {self.repr}\n"
      toString += f"  - Direcionado: {self.direcionado}\n"
      toString += f"  - Ponderado: {self.ponderado}\n"

      # Uma string da lista de vertices pra ficar mais bonito (e reutiliza-la depois)
      toString += "Vertices: "

      if len(self.vertices) == 0:
        listaVertices = ""

      else:
        listaVertices = "["
        for vertice in self.vertices:
          listaVertices += f"{vertice}, "
        listaVertices = listaVertices[:-2] + "]"

      toString += f"{listaVertices} \n"

      # Printagem da lista (se for lista):
      if self.repr == "lista":
        toString += "Lista de adjacências:\n"
        for vertice in self.listaDict:
          toString += f"{vertice}: "
          for aresta in self.listaDict[vertice]:
            toString += f"{aresta[0]} ({aresta[1]}), "
          toString = toString[:-2] + "\n"

      # Printagem da matriz (se for matriz):
      elif self.repr == "matriz":
        toString += "Matriz de adjacências:\n"

        listaVertices = listaVertices.replace("[", "")
        listaVertices = listaVertices.replace("]", "")
        listaVertices = listaVertices.replace(",", "")
        toString += f"  {listaVertices} \n"

        for i in range(len(self.matrizAdjacencias)):
          toString += f"{self.vertices[i]} "
          toString += str(self.matrizAdjacencias[i]).replace("[", "").replace(
              "]", "").replace(",", "")
          toString += "\n"

      # for i in range(50):
      #   toString += "👍"

      return toString + "\n"