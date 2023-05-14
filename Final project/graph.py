import networkx as nx
from networkx.algorithms.components import is_connected
import matplotlib.pyplot as plt
import numpy as np

def spectral_partitioning_algorithm(graph, G):
    def update_graph(graph, pos, cluster, conductance, i,color='red',min_conductance=None):
        colors = [color if node in cluster else 'blue' for node in graph.nodes()]
        plt.clf()
        plt.title(f'Iteration {i}: Conductance {conductance:.4f} (min: {min_conductance:.4f})')
        nx.draw(graph, pos=pos, with_labels=True, node_color=colors)
        plt.draw()
        plt.pause(0.5)

    n = len(G)
    D = np.diag(np.sum(G, axis=1))
    L = D - G
    
    x = np.random.rand(n)
    for i in range(100):
        x = L @ x
        x = x / np.linalg.norm(x)
        l_max = np.transpose(x) @ L @ x 
    y = np.random.rand(n)
    m = [[1/(n**0.5)] for _ in range(n)]
    for j in range(100):
        L_min = l_max * np.identity(len(L)) - L - l_max * (m @ np.transpose(m))
        y = L_min @ y
        y = y / np.linalg.norm(y)
    
    vertices = list(range(n))
    vertices.sort(key=lambda v: y[v])
    
    min_conductance = np.inf
    cluster = []

    plt.figure()
    plt.ion()

    for i in range(1, n):
        S = set(vertices[:i])
        T = set(vertices[i:])
        cut_size = np.sum(G[list(S),:][:,list(T)])
        conductance = cut_size / min(np.sum(G[list(S),:], axis=1).sum(), np.sum(G[list(T),:], axis=1).sum())
        if conductance < min_conductance:
            min_conductance = conductance
            cluster = S
        update_graph(graph, pos, set(vertices[:i]), conductance, i, color='green',min_conductance=min_conductance)
        print("Minimum conductance:", min_conductance)
        print("Cluster:", cluster)
        update_graph(graph, pos, cluster, conductance, i,min_conductance=min_conductance)
        

    plt.ioff()
    plt.show()
    plt.close()
    return min_conductance, cluster

# n is the number of vertices, p is the probability of an edge
n = 20
p = 0.1
while True:
    graph = nx.erdos_renyi_graph(n, p)
    if is_connected(graph):
        break

pos = nx.spring_layout(graph)
min_conductance, cluster = spectral_partitioning_algorithm(graph, nx.adjacency_matrix(graph).todense())





