import networkx as nx
import random
import numpy as np
import os 
from networkx.algorithms.community import greedy_modularity_communities

def facebook_circles_graph(filename, weight, save_double_edges = False):
	
	f = open(filename+'.txt', 'r')
	G = nx.Graph()

	G.add_nodes_from(np.arange(4039))

	for line in f:
		node_a,node_b = line.split()
		G.add_edges_from([(node_a, node_b)])
		G[node_a][node_b]['weight'] = weight #np.random.uniform(0,.1,1)[0]

	f.close()
	#c = list(greedy_modularity_communities(G))
	#print(f'num of communities: {len(c)}')
	#print(c[0])
	if save_double_edges:
		with open(filename + '_double_edges.txt', 'w+') as f:
			
			for v1,v2 in G.edges():
                #for it in range(edata['weight']):
				f.write('%s	%s%s'%(v1, v2, os.linesep)) 
				f.write('%s	%s%s'%(v2, v1, os.linesep)) 
	return G