import networkx as nx
import random
import os
import numpy as np

def generateGraph (n,m,filename='',pw=.75,maxw=5):
    G = nx.dense_gnm_random_graph(n,m)
    for e in G.edges():
        if random.random() < pw:
            G[e[0]][e[1]]['weight'] = 1
        else:
            G[e[0]][e[1]]['weight'] = random.randint(2,maxw)
    if filename:
        with open(filename, 'w+') as f:
            f.write('%s %s%s' %(len(G.nodes()), len(G.edges()), os.linesep))
            for v1,v2,edata in G.edges(data=True):
                for it in range(edata['weight']):
                    f.write('%s %s%s' %(v1, v2, os.linesep))
    return G

def generateGraph_ours (n,m,filename='',p_cliq =.75):
    
    #DG = nx.DiGraph()
    G = nx.Graph()
    nodes_a = []
    nodes_b = []
    for i in np.arange(n):

        toss = np.random.random_sample()
        if toss >= 0.7:
            G.add_nodes_from(i, color = 'red', active = 0, t = 0)
            nodes_a.append(i)
        else:
            G.add_nodes_from(i, color = 'blue', active = 0, t = 0)
            nodes_b.append(i)

    
    i = 0 
    while i < m:
        Y = np.random.binomial(1, p_cliq, 1)
        n_1 = np.random.randint(0,n)
        
        if n_1 in nodes_a:
            if Y == 1:
                n_2 = nodes_a[np.random.randint(0, len(nodes_a))]

            else:
                n_2 = nodes_b[np.random.randint(0, len(nodes_b))]
        else:
            if Y == 1:
                n_2 = nodes_b[np.random.randint(0, len(nodes_b))]
            else:
                n_2 = nodes_a[np.random.randint(0, len(nodes_a))]

        if G.has_edge(n_1,n_2) or n_1 == n_2:
            continue 

        G.add_edges_from([(n_1,n_2)])
        a = 0.0 
        b = 0.5
        G[n_1][n_2]['weight'] = (b-a) * np.random.random_sample() + a
        i+=1

    if filename:
        with open(filename, 'w+') as f:
            f.write('%s %s%s' %(len(G.nodes()), len(G.edges()), os.linesep))
            for v1,v2,edata in G.edges(data=True):
                for it in range(edata['weight']):
                    f.write('%s %s%s' %(v1, v2, os.linesep))
    return G 

def generateGraphNPP(n,filename='',p_with =.75, p_across = 0.1, group_ratio = 0.7):
    
    #DG = nx.DiGraph()
    G = nx.Graph()
    
    for i in np.arange(n):

        toss = np.random.uniform(0,1.0,1)[0]
       #print(i)
        if toss <= group_ratio:
            G.add_node(i, color = 'red', active = 0, t = 0)
            
        else:
            G.add_node(i, color = 'blue', active = 0, t = 0)         
    num_edges = 0 
    for i in np.arange(n):
        for j in np.arange(n):
            if G.has_edge(i,j) or i == j:
                continue 

            if G.nodes[i]['color'] == G.nodes[j]['color']:
                Y = np.random.binomial(1, p_with, 1)[0]
                if Y == 1:
                    G.add_edges_from([(i,j)])
                    G[i][j]['weight'] = np.random.uniform(0,0.1,1)[0]
                    num_edges +=1

            else:
                Y = np.random.binomial(1, p_across, 1)[0]
                if Y == 1:
                    G.add_edges_from([(i,j)])
                    G[i][j]['weight'] = np.random.uniform(0,.1,1)[0]
                    num_edges +=1 

    print(f'number of edges: {num_edges}')
        

    if filename:
        with open(filename, 'w+') as f:
            f.write('%s %s%s' %(len(G.nodes()), len(G.edges()), os.linesep))
            for n, ndata in G.nodes(data=True):
                f.write('%s %s%s'%(n, ndata['color'], os.linesep))
            for v1,v2,edata in G.edges(data=True):
                #for it in range(edata['weight']):
                f.write('%s %s %s%s'%(v1, v2, edata['weight'], os.linesep))

    return G 

if __name__ == '__main__':
    generateGraph(30, 120, 'small_graph.txt')