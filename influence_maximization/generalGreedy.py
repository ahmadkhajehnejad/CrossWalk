
from priorityQueue import PriorityQueue as PQ
from IC import *
import numpy as np
import multiprocessing
import utils as ut
import math 


def map_IC_timing(inp):

    G,S,v,gamma_a, gamma_b = inp
    R = 100
    priority = 0.0
    priority_a = 0.0
    priority_b = 0.0
    F_a = 0.0
    F_b = 0.0
    if v not in S:
        for j in range(R): # run R times Random Cascade
        # for different objective change this priority selection 
            T, T_a, T_b = runIC_fair_timings((G,S + [v], gamma_a, gamma_b))
            priority_a += float(T_a)/R
            priority_b += float(T_b)/R
            priority   += float(T_a + T_b)/R

    return (v,priority,priority_a, priority_b)

def map_IC(inp):
        G,S,p = inp
        #print(S)
        return len(runIC(G,S,p))

def map_fair_IC(inp):

        G,S = inp
        #print(S)
        R = 500
        influenced = 0.0
        influenced_grouped = {c:0 for c in np.unique([G.nodes[v]['color'] for v in G.nodes])}
        pool = multiprocessing.Pool(multiprocessing.cpu_count() * 2)
        results = pool.map(runIC_fair, [(G,S) for i in range(R)])
        pool.close()
        pool.join()
        influenced_grouped = {c:0 for c in np.unique([G.nodes[v]['color'] for v in G.nodes])}
        for T,T_grouped in results:
        #for j in range(R):
            #T, T_a, T_b = runIC_fair(G,S)
            influenced   += float(len(T)) / R
            for c,t in T_grouped.items():
                influenced_grouped[c] += float(len(t)) / R

        return (influenced, influenced_grouped)

'''
def map_fair_IC(inp):

        G,S = inp
        #print(S)
        R = 500
        influenced, influenced_a, influenced_b = (0.0,)*3
        pool = multiprocessing.Pool(multiprocessing.cpu_count() * 2)
        results = pool.map(runIC_fair, [(G,S) for i in range(R)])
        pool.close()
        pool.join()
        for T,T_a,T_b in results: 
        #for j in range(R):
            #T, T_a, T_b = runIC_fair(G,S)
            influenced   += float(len(T)) / R
            influenced_a += float(len(T_a)) / R
            influenced_b += float(len(T_b)) / R

        return (influenced, influenced_a, influenced_b)
'''

def map_select_next_seed_greedy(inp):
    # selects greedily 
        G,S,v = inp
        R = 500 # 100 #  
        priority = 0.0
        if v not in S:
            for j in range(R): # run R times Random Cascade
            # for different objective change this priority selection 
                T, T_grouped = runIC_fair((G,S + [v]))
                priority -= float(len(T))/R

        return (v,priority)


# def map_select_next_seed_greedy(inp):
#     # selects greedily 
#         G,S,v = inp
#         R = 100
#         priority = 0.0
#         if v not in S:
#             for j in range(R): # run R times Random Cascade
#             # for different objective change this priority selection 
#                 T, T_a, T_b = runIC_fair((G,S + [v]))
#                 priority -= float(len(T))/R
# 
#         return (v,priority)


def map_select_next_seed_log_greedy_prev(inp):
    # selects greedily 
        G,S,v,gamma = inp
        R = 100
        priority = 0.0
        e = 1e-20
        if v not in S:
            for j in range(R): # run R times Random Cascade
            # for different objective change this priority selection 
                T, T_a, T_b = runIC_fair((G,S + [v]))
                priority -= (math.log10(float(len(T_a)) + 1e-20) + gamma * math.log10(float(len(T_b)) + 1e-20))/R

        return (v,priority)

def map_select_next_seed_log_greedy(inp):
    # selects greedily 
        G,S,v,gamma = inp
        R = 100
        priority = 0.0
        e = 1e-20
        F_a = 0.0
        F_b = 0.0 
        if v not in S:
            for j in range(R): # run R times Random Cascade
            # for different objective change this priority selection 
                T, T_a, T_b = runIC_fair((G,S + [v]))
                F_a += float(len(T_a))/R
                F_b += float(len(T_b))/R
                #priority -= (math.log10(float(len(T_a)) + 1e-20) + gamma * math.log10(float(len(T_b)) + 1e-20))/R
            priority -= (math.log10(F_a + 1e-20) + gamma * math.log10(F_b + 1e-20))

        return (v,priority)

def map_select_next_seed_root_greedy(inp):
    # selects greedily 
        G,S,v,gamma,beta = inp
        R = 100
        priority = 0.0
        F_a = 0.0 
        F_b = 0.0
        if v not in S:
            for j in range(R): # run R times Random Cascade
            # for different objective change this priority selection 
                T, T_a, T_b = runIC_fair((G,S + [v]))
                F_a += float(len(T_a))/ R
                F_b += float(len(T_b))/ R

                #priority -= (float(len(T_a))**(1/gamma) + float(len(T_b))**(1/gamma))**beta/R
            priority -= ((F_a)**(1.0/gamma) + (F_b)**(1.0/gamma))**beta
        return (v,priority)

def map_select_next_seed_root_majority_greedy(inp):
    # selects greedily 
        G,S,v,gamma = inp
        R = 100
        priority = 0.0
        F_a = 0.0 
        F_b = 0.0
        if v not in S:
            for j in range(R): # run R times Random Cascade
            # for different objective change this priority selection 
                T, T_a, T_b = runIC_fair((G,S + [v]))
                F_a += float(len(T_a))/ R
                F_b += float(len(T_b))/ R

                #priority -= (float(len(T_a))**(1/gamma) + float(len(T_b))**(1/gamma))**beta/R
            priority -= ((F_a)**(1.0/gamma)*0 + F_b)
        return (v,priority)

def map_select_next_seed_norm_greedy(inp):
    # selects greedily 
        G,S,v,gamma = inp
        R = 100
        priority = 0.0
        if v not in S:
            for j in range(R): # run R times Random Cascade
            # for different objective change this priority selection 
                T, T_a, T_b = runIC_fair((G,S))
                priority -= ((float(len(T_a))**(1/gamma) + float(len(T_b))**(1/gamma))**gamma)/R

        return (v,priority)

def map_select_next_seed_set_cover(inp):
    # selects greedily 
        G,S,v = inp
        R = 100
        priority = 0.0
        priority_a = 0.0
        priority_b = 0.0
        if v not in S:
            for j in range(R): # run R times Random Cascade
            # for different objective change this priority selection 
                T, T_a, T_b = runIC_fair((G, S + [v]))
                priority += float(len(T))/R # not subratacting like other formulations adding a minus later 
                priority_a += float(len(T_a))/R
                priority_b += float(len(T_b))/R

        return (v,priority,priority_a,priority_b)

def generalGreedy_parallel_inf(G, k, p=.01):
    ''' Finds initial seed set S using general greedy heuristic
    Input: G -- networkx Graph object
    k -- number of initial nodes needed
    p -- propagation probability
    Output: S -- initial set of k nodes to propagate
    parallel computation of influence of the node, but, probably, since the computation is not that complex 
    '''
    #import time
    #start = time.time()
    #define map function
     #CC_parallel(G, seed_size, .01)

    #results = []#np.asarray([])
    R = 500 # number of times to run Random Cascade
    S = [] # set of selected nodes
    # add node to S if achieves maximum propagation for current chosen + this node
    for i in range(k):
        s = PQ() # priority queue

        for v in G.nodes():
            if v not in S:
                s.add_task(v, 0) # initialize spread value
                [priority, count, task] = s.entry_finder[v]
                pool = multiprocessing.Pool(multiprocessing.cpu_count()/2)
                results = pool.map(map_IC, [(G,S + [v],p)]*R)
                pool.close()
                pool.join() 
                s.add_task(v, priority - float(np.sum(results))/R)
                #for j in range(R): # run R times Random Cascade
                     #[priority, count, task] = s.entry_finder[v]
                  #  s.add_task(v, priority - float(len(runIC(G, S + [v], p)))/R) # add normalized spread value
        task, priority = s.pop_item()
        S.append(task)
        #print(i, k, time.time() - start)
    return S

def generalGreedy(G, k, p=.01):
    ''' Finds initial seed set S using general greedy heuristic
    Input: G -- networkx Graph object
    k -- number of initial nodes needed
    p -- propagation probability
    Output: S -- initial set of k nodes to propagate
    '''
    #import time
    #start = time.time()
    R = 200 # number of times to run Random Cascade
    S = [] # set of selected nodes
    # add node to S if achieves maximum propagation for current chosen + this node
    for i in range(k): # cannot parallellize
        s = PQ() # priority queue

        for i_, v in enumerate(G.nodes()):
            if v not in S:
                s.add_task(v, 0) # initialize spread value
                #[priority, count, task] = s.entry_finder[v]
                for j in range(R): # run R times Random Cascade The gain of parallelizing isn't a lot as the one runIC is not very complex maybe for huge graphs 
                    [priority, count, task] = s.entry_finder[v]
                    s.add_task(v, priority - float(len(runIC(G, S + [v], p)))/R) # add normalized spread value

        task, priority = s.pop_item()
        print(task, priority)
        S.append(task)
        #print(i, k, time.time() - start)
    return S


def generalGreedy_node_parallel(filename, G, budget, gamma, beta=1.0, type_algo=1, G_greedy = None):
    ''' Finds initial seed set S using general greedy heuristic
    Input: G -- networkx Graph object
    k -- number of initial nodes needed
    p -- propagation probability
    Output: S -- initial set of k nodes to propagate
    '''

    if G_greedy is None:
        G_greedy = G

    # import time
    # start = time.time()
    # R = 200 # number of times to run Random Cascade
    S = []  # set of selected nodes
    influenced = []
    influenced_grouped = []
    seeds = []
    seed_range = []
    if type_algo == 1:
        filename = filename + f'_greedy_'

    elif type_algo == 2:
        filename = filename + f'_log_gamma_{gamma}_'

    elif type_algo == 3:
        filename = filename + f'_root_gamma_{gamma}_beta_{beta}_'

    elif type_algo == 4:
        filename = filename + f'_root_majority_gamma_{gamma}_beta_{beta}_'

    # stats = ut.graph_stats(G, print_stats=False)

    try:

        influenced, influenced_a, influenced_b, seeds_a, seeds_b = ut.read_files(filename)

        raise Exception('It was supposed not to be reached.')

        S = seeds_a[-1] + seeds_b[-1]

        if len(S) >= budget:
            # ut.write_files(filename,influenced, influenced_a, influenced_b, seeds_a, seeds_b)
            print(influenced_a)
            print("\n\n")
            print(influenced_b)
            print(" Seed length ", len(S))

            ut.plot_influence(influenced_a, influenced_b, len(S), filename, stats['group_a'], stats['group_b'],
                              [len(S_a) for S_a in seeds_a], [len(S_b) for S_b in seeds_b])

            return (influenced, influenced_a, influenced_b, seeds_a, seeds_b)
        else:
            seed_range = range(budget - len(S))

    except FileNotFoundError:
        print(f'{filename} not Found ')

        seed_range = range(budget)

    # add node to S if achieves maximum propagation for current chosen + this node
    for i in seed_range:  # cannot parallellize
        print('--------', i)
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        # results = None
        if type_algo == 1:
            results = pool.map(map_select_next_seed_greedy, ((G_greedy, S, v) for v in G_greedy.nodes()))
        elif type_algo == 2:
            results = pool.map(map_select_next_seed_log_greedy, ((G_greedy, S, v, gamma) for v in G_greedy.nodes()))
        elif type_algo == 3:
            results = pool.map(map_select_next_seed_root_greedy, ((G_greedy, S, v, gamma, beta) for v in G_greedy.nodes()))
        elif type_algo == 4:
            results = pool.map(map_select_next_seed_root_majority_greedy, ((G_greedy, S, v, gamma) for v in G_greedy.nodes()))

        pool.close()
        pool.join()

        s = PQ()  # priority queue
        # if results == None:

        for v, priority in results:  # run R times Random Cascade The gain of parallelizing isn't a lot as the one runIC is not very complex maybe for huge graphs
            s.add_task(v, priority)

        node, priority = s.pop_item()
        S.append(node)
        I, I_grouped = map_fair_IC((G, S))
        influenced.append(I)
        influenced_grouped.append(I_grouped)
        group = G.nodes[node]['color']
        print(f'{i + 1} Selected Node is {node} group {group} I_grouped = {I_grouped}')

        S_g = {c:[] for c in np.unique([G.nodes[v]['color'] for v in G.nodes])}
        for n in S:
            c = G.nodes[n]['color']
            S_g[c].append(n)

        seeds.append(S_g)  # id's of the seeds so the influence can be recreated
        # print(i, k, time.time() - start)
    # print ( "\n \n  I shouldn't be here.   ********* \n \n ")
    # ut.plot_influence(influenced_a, influenced_b, len(S), filename, stats['group_a'], stats['group_b'],
    #                   [len(S_a) for S_a in seeds_a], [len(S_b) for S_b in seeds_b])

    ut.write_files(filename, influenced, influenced_grouped, seeds)

    return (influenced, influenced_grouped, seeds)


# def generalGreedy_node_parallel(filename, G, budget, gamma, beta = 1.0, type_algo = 1):
#     ''' Finds initial seed set S using general greedy heuristic
#     Input: G -- networkx Graph object
#     k -- number of initial nodes needed
#     p -- propagation probability
#     Output: S -- initial set of k nodes to propagate
#     '''
#     #import time
#     #start = time.time()
#     #R = 200 # number of times to run Random Cascade
#     S = [] # set of selected nodes
#     influenced = []
#     influenced_a = []
#     influenced_b = []
#     seeds_a = []
#     seeds_b = []
#     seed_range = []
#     if  type_algo == 1:
#             filename = filename + f'_greedy_'
#
#     elif type_algo == 2:
#             filename = filename + f'_log_gamma_{gamma}_'
#
#     elif type_algo == 3:
#              filename = filename + f'_root_gamma_{gamma}_beta_{beta}_'
#
#     elif type_algo == 4:
#              filename = filename + f'_root_majority_gamma_{gamma}_beta_{beta}_'
#
#
#     stats = ut.graph_stats(G, print_stats = False)
#
#     try :
#
#         influenced, influenced_a, influenced_b, seeds_a, seeds_b = ut.read_files(filename)
#         S = seeds_a[-1] + seeds_b[-1]
#
#         if len(S) >= budget:
#             #ut.write_files(filename,influenced, influenced_a, influenced_b, seeds_a, seeds_b)
#             print(influenced_a)
#             print( "\n\n")
#             print(influenced_b)
#             print(" Seed length ", len(S))
#
#             ut.plot_influence(influenced_a, influenced_b, len(S), filename , stats['group_a'], stats['group_b'], [len(S_a) for S_a in seeds_a] , [len(S_b) for S_b in seeds_b])
#
#             return (influenced, influenced_a, influenced_b, seeds_a, seeds_b)
#         else:
#             seed_range = range(budget - len(S))
#
#     except FileNotFoundError:
#         print( f'{filename} not Found ')
#
#         seed_range = range(budget)
#
#     # add node to S if achieves maximum propagation for current chosen + this node
#     for i in seed_range: # cannot parallellize
#         print('--------', i)
#         pool = multiprocessing.Pool(multiprocessing.cpu_count())
#         #results = None
#         if type_algo == 1:
#             results = pool.map(map_select_next_seed_greedy, ((G,S,v) for v in G.nodes()))
#         elif type_algo == 2:
#             results = pool.map(map_select_next_seed_log_greedy, ((G,S,v,gamma) for v in G.nodes()))
#         elif type_algo == 3:
#             results = pool.map(map_select_next_seed_root_greedy, ((G,S,v,gamma,beta) for v in G.nodes()))
#         elif type_algo == 4:
#             results = pool.map(map_select_next_seed_root_majority_greedy, ((G,S,v,gamma) for v in G.nodes()))
#
#
#         pool.close()
#         pool.join()
#
#         s = PQ() # priority queue
#         #if results == None:
#
#         for v,priority in results: # run R times Random Cascade The gain of parallelizing isn't a lot as the one runIC is not very complex maybe for huge graphs
#                     s.add_task(v, priority)
#
#         node, priority = s.pop_item()
#         S.append(node)
#         I,I_a, I_b = map_fair_IC((G,S))
#         influenced.append(I)
#         influenced_a.append(I_a)
#         influenced_b.append(I_b)
#         S_red  = []
#         S_blue = []
#         group = G.nodes[node]['color']
#         print(f'{i+1} Selected Node is {node} group {group} Ia = {I_a} Ib {I_b}')
#         for n in S:
#             if G.nodes[n]['color'] == 'red':
#                 S_red.append(n)
#             else:
#                 S_blue.append(n)
#
#         seeds_a.append(S_red) # id's of the seeds so the influence can be recreated
#         seeds_b.append(S_blue)
#         #print(i, k, time.time() - start)
#     #print ( "\n \n  I shouldn't be here.   ********* \n \n ")
#     ut.plot_influence(influenced_a, influenced_b, len(S), filename , stats['group_a'], stats['group_b'], [len(S_a) for S_a in seeds_a] , [len(S_b) for S_b in seeds_b])
#
#     ut.write_files(filename,influenced, influenced_a, influenced_b, seeds_a, seeds_b)
#
#
#     return (influenced, influenced_a, influenced_b, seeds_a, seeds_b)

def generalGreedy_node_set_cover(filename, G, budget, gamma_a = 1e-2, gamma_b = 0, type_algo = 1):
    ''' Finds initial seed set S using general greedy heuristic
    Input: G -- networkx Graph object
    k -- fraction of population needs to be influenced in both groups 
    p -- propagation probability
    Output: S -- initial set of k nodes to propagate
    '''
    #import time
    #start = time.time()
    #R = 200 # number of times to run Random Cascade
   
   
    stats = ut.graph_stats(G, print_stats = False)

    if  type_algo == 1:
            filename = filename + f'_set_cover_reach_{budget}_'
    elif type_algo == 2:
            filename = filename + f'_set_cover_timings_reach_{budget}_gamma_a_{gamma_a}_gamma_b_{gamma_b}_'
    elif type_algo == 3:
            filename = filename + f'_set_cover_timings_reach_{budget}_gamma_a_{gamma_a}_gamma_b_{gamma_a}_'

    
    reach = 0.0
    S = [] # set of selected nodes
    # add node to S if achieves maximum propagation for current chosen + this node
    influenced = []
    influenced_a = []
    influenced_b = []
    seeds_a = []
    seeds_b = []

    try :

        influenced, influenced_a, influenced_b, seeds_a, seeds_b = ut.read_files(filename)
        reach = min(influenced_a[-1]/stats['group_a'],budget) + min(influenced_b[-1]/stats['group_b'],budget)
        S = seeds_a[-1] + seeds_b[-1]
        if reach >= budget:
            #ut.write_files(filename,influenced, influenced_a, influenced_b, seeds_a, seeds_b)
            print(influenced_a)
            print( "\n\n")
            print(influenced_b)            
            print(f" reach: {reach}")
            ut.plot_influence(influenced_a, influenced_b, len(S), filename , stats['group_a'], stats['group_b'], [len(S_a) for S_a in seeds_a] , [len(S_b) for S_b in seeds_b])
            return (influenced, influenced_a, influenced_b, seeds_a, seeds_b)
        
    except FileNotFoundError:
        print( f'{filename} not Found ')
        
    i = 0
    while reach < 2*budget: # cannot parallellize 

        pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)

        if  type_algo == 1:
            results = pool.map(map_select_next_seed_set_cover, ((G,S,v) for v in G.nodes()))
        elif type_algo == 2:
            results = pool.map(map_IC_timing, ((G,S,v,gamma_a, gamma_b) for v in G.nodes()))
        elif type_algo == 3:
            results = pool.map(map_IC_timing, ((G,S,v,gamma_a, gamma_a) for v in G.nodes()))

        pool.close()
        pool.join()

        s = PQ() # priority queue
        for v,p,p_a,p_b in results: # 
            s.add_task(v, -(min(p_a/stats['group_a'],budget)+min(p_b/stats['group_b'],budget)))

        node, priority = s.pop_item()
        #priority = -priority # as the current priority is negative fraction 
        S.append(node)

        I,I_a, I_b = map_fair_IC((G,S))
        influenced.append(I)
        influenced_a.append(I_a)
        influenced_b.append(I_b)
        S_red  = []
        S_blue = []
        group = G.nodes[node]['color']
        
        for n in S:
            if G.nodes[n]['color'] == 'red':
                S_red.append(n)
            else:
                S_blue.append(n)
        
        seeds_a.append(S_red) # id's of the seeds so the influence can be recreated 
        seeds_b.append(S_blue)

        #reach += -priority both are fine 
        reach_a = I_a / stats['group_a']
        reach_b = I_b / stats['group_b']
        reach = (min(reach_a, budget) + min(reach_b, budget))

        print(f'{i+1} Node ID {node} group {group} Ia = {I_a} Ib {I_b} reach: {reach} reach_a {reach_a} reach_b {reach_b}')
        #print(i, k, time.time() - start)
        i+=1

    ut.plot_influence(influenced_a, influenced_b, len(S), filename , stats['group_a'], stats['group_b'], [len(S_a) for S_a in seeds_a] , [len(S_b) for S_b in seeds_b])

    ut.write_files(filename,influenced, influenced_a, influenced_b, seeds_a, seeds_b)

    return (influenced, influenced_a, influenced_b, seeds_a, seeds_b)



