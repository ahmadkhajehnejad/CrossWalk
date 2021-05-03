''' Independent cascade model for influence propagation
'''
import numpy as np 

def runIC (G, S, p = .01):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    '''
    from copy import deepcopy
    from random import random
    T = deepcopy(S) # copy already selected nodes

    # ugly C++ version
    i = 0
    while i < len(T):
        for v in G[T[i]]: # for neighbors of a selected node
            if v not in T: # if it wasn't selected yet
                w = G[T[i]][v]['weight'] # count the number of edges between two nodes
                if random() <= 1 - (1-p)**w: # if at least one of edges propagate influence
                    #print(T[i], 'influences', v)
                    T.append(v)
        i += 1

    # neat pythonic version
    # legitimate version with dynamically changing list: http://stackoverflow.com/a/15725492/2069858
    # for u in T: # T may increase size during iterations
    #     for v in G[u]: # check whether new node v is influenced by chosen node u
    #         w = G[u][v]['weight']
    #         if v not in T and random() < 1 - (1-p)**w:
    #             T.append(v)
    return T


def runIC_fair(inp):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    '''
    G, S = inp
    from copy import deepcopy
    from random import random
    T = deepcopy(S)  # copy already selected nodes
    T_grouped = {c:[] for c in np.unique([G.nodes[v]['color'] for v in G.nodes])}
    for t in T:
        c = G.nodes[t]['color']
        T_grouped[c].append(t)

    # ugly C++ version
    i = 0
    while i < len(T):
        for v in G[T[i]]:  # for neighbors of a selected node
            if v not in T:  # if it wasn't selected yet
                w = G[T[i]][v]['weight']  # probability of infection

                # r = random()
                # print(r, w)
                # if random() <= 1 - (1-p)**w: # if at least one of edges propagate influence
                if random() <= w:  # i.e. now w is actually probability and there is only one edge between two nodes
                    # print(T[i], 'influences', v)
                    T.append(v)
                    c = G.nodes[v]['color']
                    T_grouped[c].append(v)
        i += 1
    # neat pythonic version
    # legitimate version with dynamically changing list: http://stackoverflow.com/a/15725492/2069858
    # for u in T: # T may increase size during iterations
    #     for v in G[u]: # check whether new node v is influenced by chosen node u
    #         w = G[u][v]['weight']
    #         if v not in T and random() < 1 - (1-p)**w:
    #             T.append(v)
    return (T, T_grouped)


# def runIC_fair (inp):
#     ''' Runs independent cascade model.
#     Input: G -- networkx graph object
#     S -- initial set of vertices
#     p -- propagation probability
#     Output: T -- resulted influenced set of vertices (including S)
#     '''
#     G, S = inp
#     from copy import deepcopy
#     from random import random
#     T = deepcopy(S) # copy already selected nodes
#     T_a = []
#     T_b = []
#     for t in T:
#         if G.nodes[t]['color'] == 'red':
#             T_a.append(t)
#         else:
#             T_b.append(t)
#
#     # ugly C++ version
#     i = 0
#     while i < len(T):
#         for v in G[T[i]]: # for neighbors of a selected node
#             if v not in T: # if it wasn't selected yet
#                 w = G[T[i]][v]['weight'] # probability of infection
#
#                 #r = random()
#                 #print(r, w)
#                 # if random() <= 1 - (1-p)**w: # if at least one of edges propagate influence
#                 if random() <= w: # i.e. now w is actually probability and there is only one edge between two nodes
#                     #print(T[i], 'influences', v)
#                     T.append(v)
#                     if G.nodes[v]['color'] == 'red':
#                         T_a.append(v)
#                     else:
#                         T_b.append(v)
#         i += 1
#     # neat pythonic version
#     # legitimate version with dynamically changing list: http://stackoverflow.com/a/15725492/2069858
#     # for u in T: # T may increase size during iterations
#     #     for v in G[u]: # check whether new node v is influenced by chosen node u
#     #         w = G[u][v]['weight']
#     #         if v not in T and random() < 1 - (1-p)**w:
#     #             T.append(v)
#     return (T,T_a,T_b)


def runIC_fair_timings (inp):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    '''
    G, S, gamma_a, gamma_b = inp 
    
    from random import random

    T = []
    T_a = []
    T_b = []

    T_ret =   0.0 
    T_a_ret = 0.0
    T_b_ret = 0.0
    
    for s in S:
        T.append((s,0))
        if G.nodes[s]['color'] == 'red':
            T_a.append((s,0))
        else:
            T_b.append((s,0))

    i = 0
    while i < len(T):
        for v in G[T[i][0]]: # for neighbors of a selected node where T[i] = (v,t) hence T[i][0] = v
            
            w = G[T[i][0]][v]['weight'] # probability of infection 
            if random() <= w: # i.e. now w is actually probability and there is only one edge between two nodes 
                # if the the node is already infected or not 
                index = [idx for idx, item in enumerate(T) if item[0] == v] # i.e. get the timing of infection if the node is infected 
                if len(index) == 0: # if it wasn't selected yet
                    T.append((v,T[i][1]+1)) # add and increase timings 
                         
                elif (T[i][1]+1 < T[index[0]][1]):
                    T[index[0]][1] = T[i][1]+1 # select the smallest timings
        i += 1

    for n,t in T:
        #gamma_a = 1
        T_ret += (gamma_a ** t)
        if G.nodes[n]['color'] == 'red':
            T_a_ret += (gamma_a ** t)
            T_a.append((n,t))
        else:
            T_a_ret += (gamma_b ** t)
            T_b.append((n,t))

    #return (T,T_a,T_b)
    return (T_ret, T_a_ret, T_b_ret)


def runIC2(G, S, p=.01):
    ''' Runs independent cascade model (finds levels of propagation).
    Let A0 be S. A_i is defined as activated nodes at ith step by nodes in A_(i-1).
    We call A_0, A_1, ..., A_i, ..., A_l levels of propagation.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    '''
    from copy import deepcopy
    import random
    T = deepcopy(S)
    Acur = deepcopy(S)
    Anext = []
    i = 0
    while Acur:
        values = dict()
        for u in Acur:
            for v in G[u]:
                if v not in T:
                    w = G[u][v]['weight']
                    if random.random() < 1 - (1-p)**w:
                        Anext.append((v, u))
        Acur = [edge[0] for edge in Anext]
        print(i, Anext)
        i += 1
        T.extend(Acur)
        Anext = []
    return T
    
def avgSize(G,S,p,iterations):
    avg = 0
    for i in range(iterations):
        avg += float(len(runIC(G,S,p)))/iterations
    return avg



if __name__ == '__main__':

    G,S,v = inp
    R = 100
    priority = 0.0
    if v not in S:
        for j in range(R): # run R times Random Cascade
        # for different objective change this priority selection 
            T, T_a, T_b = runIC_fair(G,S + [v])
            priority -= float(len(T))/R
    #return (v,priority)


