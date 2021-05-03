import numpy as np
import networkx as nx
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from operator import add
from generateGraph import generateGraphNPP
import os
from networkx.algorithms import community
import numpy as np
from sklearn_extra.cluster import KMedoids
from load_facebook_graph import *

def load_graph(filename, p_with, p_across,  group_ratio ,num_nodes ):
	try:
		f = open(filename+'.txt', 'r')
		print("loaded: " + filename )
		G = nx.Graph()
		n,m = map(int, f.readline().split())
		for i, line in enumerate(f):
			if i < n :
				node_str = line.split()
				u = int(node_str[0])
				color = node_str[1]
				G.add_node(u, color = color)
			else:
				edges_str = line.split()
				u = int(edges_str[0])
				v = int(edges_str[1])
				weight = float(edges_str[2])
				G.add_edge(u,v, weight = weight)
		f.close()       
    # Store configuration file values
	except FileNotFoundError:
		print(f"File not found at {filename}, building new graph...")
		G = generateGraphNPP(num_nodes, filename = filename +'.txt', p_with = p_with, p_across = p_across, group_ratio =group_ratio)

	return G

def graph_stats(G, print_stats =True):

	# print average weights of each group
	w_a_within, w_across, w_b_within, num_a, num_b, edges_a, edges_b, edges_across = (0.0,)*8
	for n, nbrs in G.adj.items():
		color = G.nodes[n]['color']
		# color = G.node[n]['color']
		if color == 'red':
			num_a +=1
		else:
			num_b +=1

		for nbr, eattr in nbrs.items():
			if G.nodes[nbr]['color'] == color:
			# if G.node[nbr]['color'] == color:
				if color == 'red':
					w_a_within += eattr['weight']
					edges_a +=1
				else:
					w_b_within  += eattr['weight']
					edges_b +=1

			else :
				w_across += eattr['weight']
				edges_across += 1

	#for v1,v2,edata in G.edges(data=True):
	stats = {}
	stats['total_nodes'] = int(num_a + num_b)
	stats['group_a'] = int(num_a)
	stats['group_b'] = int(num_b)
	stats['total_edges'] = int(edges_a / 2 + edges_b / 2 +edges_across / 2)
	stats['edges_group_a'] = int(edges_a / 2)
	stats['edges_group_b'] = int(edges_b / 2)
	stats['edges_across'] = int(edges_across / 2)
	stats['weights_group_a'] = w_a_within / edges_a
	stats['weights_group_b'] = w_b_within / edges_b
	stats['weights_across'] = w_across / edges_across
	if print_stats:
		print(f'\n \n Red Nodes: {num_a}, Blue Nodes: {num_b}, edges total = {edges_a / 2 + edges_b / 2 +edges_across / 2} edges_within a: {edges_a / 2}, edges_within_b {edges_b / 2}, edges_across {edges_across / 2}, average degree a: {edges_a / num_a}, average degree b: {edges_b / num_b}, weights within red {w_a_within / edges_a}, weights within b: {w_b_within / edges_b}, weights accross: {w_across / edges_across} \n \n \n ')
	return stats 

def write_files(filename, num_influenced, num_influenced_grouped, seeds):
	'''
	write num_influenced, num_influenced_a, num_influenced_b -> list
		  and seeds_a list of lists i.e. actual id's of the seeds chosen
		      seeds_b list of lists
	each row
	I I_a I_b [seed_list_a comma separated];[seed_list_b]
	.
	.
	.
	'''
	f = open(filename +'_results.txt', 'w')
	for i in range(len(num_influenced)):
		I = num_influenced[i]
		f.write(f'{str(I)} ')
		for c in num_influenced_grouped[i]:
			f.write(f'{str(c)} {str(num_influenced_grouped[i][c])} ')

		for c in seeds[i]:
			f.write(f'{str(c)} ')
			for j, s in enumerate(seeds[i][c]):
				if j == len(seeds[i][c]) - 1:
					f.write(f'{str(s)}; ')
				else:
					f.write(f'{str(s)} ')
		f.write('\n')

	f.close()


# def write_files(filename, num_influenced, num_influenced_a, num_influenced_b, seeds_a, seeds_b):
# 	'''
# 	write num_influenced, num_influenced_a, num_influenced_b -> list
# 		  and seeds_a list of lists i.e. actual id's of the seeds chosen
# 		      seeds_b list of lists
# 	each row
# 	I I_a I_b [seed_list_a comma separated];[seed_list_b]
# 	.
# 	.
# 	.
# 	'''
# 	f = open(filename +'_results.txt', 'w')
# 	for I,I_a,I_b,S_a,S_b in zip(num_influenced,num_influenced_a,num_influenced_b,seeds_a,seeds_b):
# 		f.write(f'{str(I)} {str(I_a)} {str(I_b)} ')
# 		for i,seed in enumerate(S_a):
# 			if i == len(S_a) - 1:
# 				f.write(f'{seed}')
# 			else:
# 				f.write(f'{seed},')
# 		f.write(';')
# 		for i,seed in enumerate(S_b):
# 			if i == len(S_b) - 1:
# 				f.write(f'{seed}')
# 			else:
# 				f.write(f'{seed},')
# 		f.write('\n')
# 	f.close()

def read_files(filename):
	'''
	returns num_influenced, num_influenced_a, num_influenced_b -> list 
		  and seeds_a list of lists i.e. actual id's of the seeds chosen 
		      seeds_b list of lists

	'''
	f = open(filename+'_results.txt', 'r')
	num_influenced =[]
	num_influenced_a =[]
	num_influenced_b =[]
	seeds_a =[]
	seeds_b =[]

	for line in f:
		I, I_a, I_b, residue = line.split()
		num_influenced.append(float(I))
		num_influenced_a.append(float(I_a))
		num_influenced_b.append(float(I_b))

		S_a,S_b = residue.split(';')
		S_a_list = []
		
		if S_a != '':
			S_a_list = list(map(int,S_a.split(',')))
		seeds_a.append(S_a_list)

		S_b_list = []
		if S_b != '':
			S_b_list = list(map(int,S_b.split(',')))
		seeds_b.append(S_b_list)


	f.close()
	return num_influenced, num_influenced_a, num_influenced_b, seeds_a, seeds_b

def plot_influence(influenced_a, influenced_b, num_seeds, filename , population_a, population_b, num_seeds_a , num_seeds_b):

	
	# total influence
	fig = plt.figure(figsize=(6,4))
	plt.plot(np.arange(1, num_seeds + 1),list(map(add,influenced_a,influenced_b)),'g+')
	plt.xlabel('Number of Seeds')
	plt.ylabel('Total Influenced Nodes')
	#plt.legend(loc='best')
	plt.savefig(filename + '_total_influenced.png',bbox_inches='tight')
	plt.close(fig)
	# total influence fraction 
	fig = plt.figure(figsize=(6,4))
	plt.plot(np.arange(1, num_seeds + 1),np.asarray(list(map(add,influenced_a,influenced_b)))/(population_a + population_b),'g+')
	plt.xlabel('Number of Seeds')
	plt.ylabel('Total Fraction Influenced Nodes')
	#plt.legend(loc='best')
	plt.savefig(filename + '_total_fraction_influenced.png',bbox_inches='tight')
	plt.close(fig)
	# group wise influenced
	fig = plt.figure(figsize=(6,4))
	plt.plot(np.arange(1, num_seeds + 1), influenced_a, 'r+', label='Group A')
	plt.plot(np.arange(1, num_seeds + 1), influenced_b,'b.', label='Group B')
	plt.xlabel('Number of Seeds')
	plt.ylabel('Total Influenced Nodes')
	plt.legend(loc='best')
	plt.savefig(filename + '_group_influenced.png',bbox_inches='tight')
	plt.close(fig)

	# fraction group influenced 
	fig = plt.figure(figsize=(6,4))
	plt.plot(np.arange(1, num_seeds + 1), np.asarray(influenced_a) / population_a, 'r+', label='Group A')
	plt.plot(np.arange(1, num_seeds + 1), np.asarray(influenced_b) / population_b,'b.', label='Group B')
	plt.xlabel('Number of Seeds')
	plt.ylabel('Fraction of Influenced Nodes')
	plt.legend(loc='best')
	plt.savefig(filename + '_fraction_group_influenced.png',bbox_inches='tight')
	plt.close(fig)

	# Seeds group memeber ship 
	#fig = plt.figure(figsize=(6,4))
	fig, ax = plt.subplots(figsize=(8,4))
	bar_width = 0.35
	index = np.arange(1, num_seeds + 1)
	rects1 = ax.bar(index, num_seeds_a, bar_width,
                 color='r',
                label='Group A')
	print(num_seeds_b)
	rects2 = ax.bar(index + bar_width, num_seeds_b , bar_width,
	                 color='b',
	                label='Group B')
	plt.legend(loc='best')
	ax.set_xlabel('Total Number of Seeds')
	ax.set_ylabel('Number of Seeds from each group')
	ax.set_title('Seed distribution in groups')
	ax.set_xticks(index)# + bar_width / 2)

	# plt.plot(np.arange(1, num_seeds + 1), num_seeds_a / num_seeds, 'r+')
	# plt.plot(np.arange(1, num_seeds + 1), num_seeds_b / num_seeds, 'b.')
	# plt.xlabel('Total Number of Seeds')
	# plt.ylabel('Number from each group')
	plt.savefig(filename + '_seed_groups.png', bbox_inches='tight')
	plt.close(fig)	#

def plot_influence_diff(influenced_a_list, influenced_b_list, num_seeds, labels, filename, population_a, population_b):
	'''
	list of lists influenced_a and influenced_b
	'''
	fig, ax = plt.subplots(figsize=(8, 6), dpi= 80)
	index = np.arange(1, num_seeds + 1)
	for i, (influenced_a, influenced_b) in enumerate(zip (influenced_a_list, influenced_b_list)):
		ax.plot(index, (np.asarray(influenced_a) + np.asarray(influenced_b))/(population_a + population_b), label=labels[i], ls= '-', alpha=0.5)
		

	legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
	plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


	
	plt.xlabel('Number of Seeds')
	plt.ylabel('Fraction of Influenced Nodes (F(S))')
	plt.savefig(filename+'_total_influenced.png',bbox_inches='tight')
	plt.close(fig)

	# comparison abs difference 
	fig, ax = plt.subplots(figsize=(8, 6), dpi= 80)
	index = np.arange(1, num_seeds + 1)
	for i, (influenced_a, influenced_b) in enumerate(zip(influenced_a_list, influenced_b_list)):
		ax.plot(index, np.abs(np.asarray(influenced_a)/population_a - np.asarray(influenced_b)/population_b), label=labels[i], ls= '-', alpha=0.5)
		

	legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
	plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


	
	plt.xlabel('Number of Seeds')
	plt.ylabel('Absolute difference of Influenced Nodes (|Fa - Fb|)')
	plt.savefig(filename+'_difference_total_influenced.png',bbox_inches='tight')
	plt.close(fig)

def load_random_graph(filename, n,p,w):
	#return get_random_graph(filename+'.txt', n,p,w)
	try:
		f = open(filename+'.txt', 'r')
		G = nx.Graph()
		n,m = map(int, f.readline().split())
		print("loaded: " + filename )
		for i, line in enumerate(f):
			if i < n :
				node_str = line.split()
				u = int(node_str[0])
				color = node_str[1]
				G.add_node(u, color = color)
			else:
				edges_str = line.split()
				u = int(edges_str[0])
				v = int(edges_str[1])
				weight = float(edges_str[2])
				G.add_edge(u,v, weight = weight)
		f.close()   
	except FileNotFoundError:
		print(f"File not found at {filename}, building new graph...")
		G = get_random_graph(filename+'.txt', n,p,w)
	return G 

def save_graph(filename,G):
	if filename:
		with open(filename, 'w+') as f:
			f.write('%s %s%s' %(len(G.nodes()), len(G.edges()), os.linesep))
			for n, ndata in G.nodes(data=True):
				f.write('%s %s%s'%(n, ndata['color'], os.linesep))
			for v1,v2,edata in G.edges(data=True):
                #for it in range(edata['weight']):
				f.write('%s %s %s%s'%(v1, v2, edata['weight'], os.linesep))
		print("saved")

def get_random_graph(filename,n,p,w):
	
	G = nx.binomial_graph(n, p)
	color = 'blue' # all nodes are one color 
	nx.set_node_attributes(G, color, 'color')
	nx.set_edge_attributes(G, w, 'weight')
	
	#save_graph(filename, G)

	return G 


def get_twitter_data(filename,w = None, save = False	):
	'''
	reads twitter data, makes bipartition and assign group memebership 
	with constant weights of infection 
	'''

	f = None
	DG = None
	try:
		f = open(filename+'.txt', 'r')
		print("loaded: " + filename )
		DG = nx.DiGraph()
		n,m = map(int, f.readline().split())
		for i, line in enumerate(f):
			if i < n :
				node_str = line.split()
				u = int(node_str[0])
				color = node_str[1]
				DG.add_node(u, color = color)
			else:
				edges_str = line.split()
				u = int(edges_str[0])
				v = int(edges_str[1])
				weight = float(edges_str[2])
                                #if w is not None:
				if False: 
					DG.add_edge(u,v, weight = w)
				else: 
					DG.add_edge(u,v, weight = weight)
		f.close() 
		
	except FileNotFoundError:
		#
		print(" Making graph ") 
		f = open('twitter/twitter_combined.txt', 'r')
		DG = nx.DiGraph()

		for line in f:
		    node_a, node_b = line.split()
		    DG.add_nodes_from([node_a,node_b])
		    DG.add_edges_from([(node_a, node_b)])

		    DG[node_a][node_b]['weight'] = w 
		
		print("done with edges and weights ")

		G_a , G_b = community.kernighan_lin_bisection(DG.to_undirected())
		for n in G_a:
			DG.nodes[n]['color'] = 'red'
		for n in G_b:
			DG.nodes[n]['color'] = 'blue'

		save_graph(filename, DG)
	 

	return DG


def get_facebook_data(filename,w = None, save = False):
	'''
	reads twitter data, makes bipartition and assign group memebership 
	with constant weights of infection 
	'''
	f = None
	G = None
	try:
		f = open(filename+'_with_communities.txt', 'r')
		print("loaded: " + filename )
		G = nx.Graph()
		n,m = map(int, f.readline().split())
		for i, line in enumerate(f):
			if i < n :
				node_str = line.split()
				u = int(node_str[0])
				color = node_str[1]
				G.add_node(u, color = color)
			else:
				edges_str = line.split()
				u = int(edges_str[0])
				v = int(edges_str[1])
				weight = float(edges_str[2])
				if w is not None:
					G.add_edge(u,v, weight = w)
				else: 
					G.add_edge(u,v, weight = weight)
		f.close() 
		
	except FileNotFoundError:
		#
		print(" Making graph ") 

		G = facebook_circles_graph(filename, w)

		G_a , G_b = community.kernighan_lin_bisection(G.to_undirected())
		for n in G_a:
			G.nodes[n]['color'] = 'red'
		for n in G_b:
			G.nodes[n]['color'] = 'blue'

		save_graph(filename, G)

	return G


def get_data(filename, w):
	'''
	reads rice data and assigns group memeberships
	'''

	DG = nx.DiGraph()
	print("loading: " + filename)
	with open(filename + '.attr', 'r') as f_attr:
		for line in f_attr:
			node_str = line.split()
			u = int(node_str[0])
			#if node_str[1] in ['0', '1']:
			#color = 'red' if node_str[1] == '1' else 'blue'
			color = node_str[1]
			DG.add_node(u, color=color)
	#cnt = dict()
	with open(filename + '.links', 'r') as f_edges:
		for line in f_edges:
			edges_str = line.split()
			u = int(edges_str[0])
			v = int(edges_str[1])

			# print(u, '-->', v)

			#t_1 = min(u, v)
			#t_2 = max(u, v)
			#if (t_1, t_2) in cnt:
			#	cnt[(t_1, t_2)] += 1
			#else:
			#	cnt[(t_1, t_2)] = 1

			if (u not in DG.nodes()) or (v not in DG.nodes()):
				# print('-------------', edges_str)
				continue
			DG.add_edge(u, v, weight = w)
			DG.add_edge(v, u, weight = w)

	#print('$$$$$$$$$$', np.unique(list(cnt.values())), len(cnt))
	return DG

def load_embeddings(filename, nodes):

	with open(filename, 'r') as f:

		_, d = map(int, f.readline().split())
		v = []
		em = []

		for line in f:
			node_str = line.split()
			v_ = int(node_str[0])
			if v_ not in nodes:
				continue
			v.append(v_)
			em.append([float(node_str[j]) for j in range(1, d+1)])

		return np.array(v), np.array(em)

def get_kmedoids_centers(em, k, v):
	c = KMedoids(k).fit(em).medoid_indices_
	return v[c].tolist()


def make_weighted_graph(G, weights_file, type_algo):
	W_G = nx.DiGraph()

	w_dict = dict()
	with open(weights_file, 'r') as fin:
		for line in fin:
			s = line.split()
			u, v, w = int(s[0]), int(s[1]), float(s[2])
			if u not in w_dict:
				w_dict[u] = dict()
			w_dict[u][v] = w

	for u in G.nodes:
		W_G.add_node(u, color=G.nodes[u]['color'])

	if type_algo == 1:
		for e in G.edges:
			W_G.add_edge(e[0], e[1], weight = G.edges[e]['weight'] * w_dict[e[0]][e[1]])
	elif type_algo == 2:
		mx = dict()
		for u in G.nodes:
                        if u in w_dict:
                            mx[u] = np.max(list(w_dict[u].values()))
		for e in G.edges:
			W_G.add_edge(e[0], e[1], weight = G.edges[e]['weight'] * w_dict[e[0]][e[1]] / mx[e[0]])
	else:
		raise Exception('type_algo ' + str(type_algo) + ' not supported.')

	return W_G
