''' File for testing different files in parallel
'''

import networkx as nx

from IC import runIC, avgSize
from CCparallel import CC_parallel
from generalGreedy import *
import multiprocessing
from heapq import nlargest
from generateGraph import generateGraphNPP
import utils as ut
#import matplotlib.pylab as plt
import os



if __name__ == '__main__':
    import time
    start = time.time()

    num_nodes = 500
    p_with =.025
    p_acrosses = [ 0.001]#, 0.025, 0.015, 0.005]
    #p_across =.001  #0.001
    influenced_list =[]
    influenced_a_list = []
    influenced_b_list = []
    labels = []
    seed_size = 30
    for p_across in p_acrosses:
        group_ratios = [0.7]#,0.5,0.55, 0.6, 0.65]
        for group_ratio in group_ratios:
        #group_ratio = 0.5 #0.7 
            
           
            filename=f'results/synthetic_data_{num_nodes}_{p_with}_{p_across}_{group_ratio}'
            
            # read in graph
            G = ut.load_graph(filename, p_with, p_across,  group_ratio ,num_nodes)
            
            ut.graph_stats(G)
            
            
            

             
            gammas = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5]
            #gamma = 2.5
            types = [1,2]
            for t in types:
                if t == 1:
                    gammas = [1.0]#, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5]
                elif t == 2:
                    gammas = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5]

                for gamma in gammas:
                    influenced, influenced_a, influenced_b = generalGreedy_node_parallel(filename, G, seed_size, gamma, type_algo = t)
                    influenced_a_list.append(influenced_a)
                    influenced_b_list.append(influenced_b)
                    if t == 1:
                        label = "Greedy"
                    elif t ==2:
                        label = f'Log_gamma{gamma}'

                    labels.append(label)

    filename = "results/greedy_and_log"
    stats = ut.graph_stats(G, print_stats = False)
    ut.plot_influence_diff(influenced_a_list, influenced_b_list, seed_size, labels, filename,stats['group_a'], stats['group_b'] )       

    print('Total time:', time.time() - start)
    
