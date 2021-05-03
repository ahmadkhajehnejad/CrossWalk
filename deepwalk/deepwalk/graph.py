#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Graph utilities."""

import logging
import sys
from io import open
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
import random
from random import shuffle
from itertools import product,permutations
from scipy.io import loadmat
from scipy.sparse import issparse
import numpy as np
import multiprocessing
import pickle


logger = logging.getLogger("deepwalk")

__author__ = "Bryan Perozzi"
__email__ = "bperozzi@cs.stonybrook.edu"

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

class Graph(defaultdict):
  """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""  
  def __init__(self):
    super(Graph, self).__init__(list)
    self.edge_weights = None
    self.attr = None
    # self.border_score = None
    self.border_distance = None

  def nodes(self):
    return self.keys()

  def adjacency_iter(self):
    return self.iteritems()

  def subgraph(self, nodes={}):
    subgraph = Graph()
    
    for n in nodes:
      if n in self:
        subgraph[n] = [x for x in self[n] if x in nodes]
        
    return subgraph

  def make_undirected(self):
  
    t0 = time()

    for v in list(self):
      for other in self[v]:
        if v != other:
          self[other].append(v)
    
    t1 = time()
    logger.info('make_directed: added missing edges {}s'.format(t1-t0))

    self.make_consistent()
    return self

  def make_consistent(self):
    t0 = time()
    for k in iterkeys(self):
      self[k] = list(sorted(set(self[k])))
    
    t1 = time()
    logger.info('make_consistent: made consistent in {}s'.format(t1-t0))

    self.remove_self_loops()

    return self

  def remove_self_loops(self):

    removed = 0
    t0 = time()

    for x in self:
      if x in self[x]: 
        self[x].remove(x)
        removed += 1
    
    t1 = time()

    logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1-t0)))
    return self

  def check_self_loops(self):
    for x in self:
      for y in self[x]:
        if x == y:
          return True
    
    return False

  def has_edge(self, v1, v2):
    if v2 in self[v1] or v1 in self[v2]:
      return True
    return False

  def degree(self, nodes=None):
    if isinstance(nodes, Iterable):
      return {v:len(self[v]) for v in nodes}
    else:
      return len(self[nodes])

  def order(self):
    "Returns the number of nodes in the graph"
    return len(self)    

  def number_of_edges(self):
    "Returns the number of nodes in the graph"
    return sum([self.degree(x) for x in self.keys()])/2

  def number_of_nodes(self):
    "Returns the number of nodes in the graph"
    return self.order()

  def random_walk(self, path_length, p_modified, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.

        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(list(G.keys()))]
    modified = np.random.rand() < p_modified
    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        if rand.random() >= alpha:
          if not modified:
            path.append(rand.choice(G[cur]))
          elif G.edge_weights is None:
            path.append(rand.choice(G[cur]))
          elif isinstance(G.edge_weights, str) and (G.edge_weights.startswith('prb_')):
            tmp = G.edge_weights.split('_')
            p_rb, p_br = float(tmp[1]), float(tmp[3])
            l_1 = [u for u in G[cur] if G.attr[u] == G.attr[cur]]
            l_2 = [u for u in G[cur] if G.attr[u] != G.attr[cur]]
            if (len(l_1) == 0) or (len(l_2) == 0):
              path.append(rand.choice(G[cur]))
            else:
              p = p_rb if G.attr[cur] == 1 else p_br
              if np.random.rand() < p:
                path.append(rand.choice(l_2))
              else:
                path.append(rand.choice(l_1))
          elif isinstance(G.edge_weights, str) and G.edge_weights.startswith('pch_'):
            p_ch = float(G.edge_weights.split('_')[1])
            if G.border_distance[cur] == 1:
              l_1 = [u for u in G[cur] if G.attr[u] == G.attr[cur]]
              l_2 = [u for u in G[cur] if G.attr[u] != G.attr[cur]]
            else:
              l_1 = [u for u in G[cur] if G.border_distance[u] >= G.border_distance[cur]]
              l_2 = [u for u in G[cur] if G.border_distance[u] < G.border_distance[cur]]
            if (len(l_1) == 0) or (len(l_2) == 0):
              path.append(rand.choice(G[cur]))
            else:
              if np.random.rand() < p_ch:
                path.append(rand.choice(l_2))
              else:
                path.append(rand.choice(l_1))
          elif isinstance(G.edge_weights, str) and G.edge_weights == 'random':
            path.append(rand.choice([v for v in G]))
          elif isinstance(G.edge_weights, str) and G.edge_weights.startswith('smartshortcut'):
            p_sc = float(G.edge_weights.split('_')[1])
            if np.random.rand() < p_sc:
              l_1 = [u for u in G[cur] if G.attr[u] != G.attr[cur]]
              if len(l_1) == 0:
                l_1 = [v for v in G if G.attr[v] != G.attr[cur]]
              path.append(rand.choice(l_1))
            else:
              path.append(rand.choice(G[cur]))
          else:
            path.append(np.random.choice(G[cur], 1, p=G.edge_weights[cur])[0])
        else:
          path.append(path[0])
      else:
        break
    return [str(node) for node in path]

# TODO add build_walks in here

def build_deepwalk_corpus(G, num_paths, path_length, p_modified, alpha=0,
                      rand=random.Random(0)):
  walks = []

  nodes = list(G.nodes())
  
  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
      walks.append(G.random_walk(path_length, p_modified=p_modified, rand=rand, alpha=alpha, start=node))
  
  return walks

def build_deepwalk_corpus_iter(G, num_paths, path_length, p_modified, alpha=0,
                      rand=random.Random(0)):
  walks = []

  nodes = list(G.nodes())

  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
      yield G.random_walk(path_length, p_modified=p_modified, rand=rand, alpha=alpha, start=node)


def clique(size):
    return from_adjlist(permutations(range(1,size+1)))


# http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def parse_adjacencylist(f):
  adjlist = []
  for l in f:
    if l and l[0] != "#":
      introw = [int(x) for x in l.strip().split()]
      row = [introw[0]]
      row.extend(set(sorted(introw[1:])))
      adjlist.extend([row])
  
  return adjlist

def parse_adjacencylist_unchecked(f):
  adjlist = []
  for l in f:
    if l and l[0] != "#":
      adjlist.extend([[int(x) for x in l.strip().split()]])
  
  return adjlist

def load_adjacencylist(file_, undirected=False, chunksize=10000, unchecked=True):

  if unchecked:
    parse_func = parse_adjacencylist_unchecked
    convert_func = from_adjlist_unchecked
  else:
    parse_func = parse_adjacencylist
    convert_func = from_adjlist

  adjlist = []

  t0 = time()
  
  total = 0 
  with open(file_) as f:
    for idx, adj_chunk in enumerate(map(parse_func, grouper(int(chunksize), f))):
      adjlist.extend(adj_chunk)
      total += len(adj_chunk)
  
  t1 = time()

  logger.info('Parsed {} edges with {} chunks in {}s'.format(total, idx, t1-t0))

  t0 = time()
  G = convert_func(adjlist)
  t1 = time()

  logger.info('Converted edges to graph in {}s'.format(t1-t0))

  if undirected:
    t0 = time()
    G = G.make_undirected()
    t1 = time()
    logger.info('Made graph undirected in {}s'.format(t1-t0))

  return G 


def load_edgelist(file_, undirected=True, attr_file_name=None, test_links_ratio=0., test_links_file=None, train_links_file=None):

  G = Graph()

  if attr_file_name is not None:
    G.attr = dict()
    with open(attr_file_name) as f:
      for l in f:
        id, a = l.strip().split()
        id = int(id)
        a = int(a)
        # if a in [0, 1]:
        G.attr[id] = a

    print('All attributes: ', np.unique(list(G.attr.values())))

  if (test_links_file is not None) and (train_links_file is not None) and path.isfile(test_links_file) and path.isfile(train_links_file):
    with open(train_links_file, 'rb') as fin:
        train_links = pickle.load(fin)
    for l in train_links:
        if l[4] == 1:
            G[l[0]].append(l[1])
            if undirected:
                G[l[1]].append(l[0])
  else:
    pos_test_links = []
    pos_train_links = []

    with open(file_) as f:
        for l in f:
            x, y = l.strip().split()[:2]
            x = int(x)
            y = int(y)
            if (x not in G.attr) or (y not in G.attr):
                continue
            if np.random.rand() < test_links_ratio:
                pos_test_links.append([x, y, G.attr[x], G.attr[y], 1])
            else:
                G[x].append(y)
                if undirected:
                    G[y].append(x)
                pos_train_links.append([x, y, G.attr[x], G.attr[y], 1])

    if train_links_file is not None and test_links_file is not None:
        mark_pos_links = set([(l[0],l[1]) for l in pos_train_links + pos_test_links])
        mark_neg_links = set()

        neg_test_links = []
        for l in pos_test_links:
            while True:
                x = np.random.choice([v for v in G if G.attr[v] == l[2]])
                y = np.random.choice([v for v in G if v != x and G.attr[v] == l[3]])
                if ((x,y) not in mark_pos_links) and ((y,x) not in mark_pos_links) and \
                        ((x,y) not in mark_neg_links) and ((y,x) not in mark_neg_links):
                    break
            mark_neg_links.update([(x,y)])
            neg_test_links.append([x, y, G.attr[x], G.attr[y], 0])

        neg_train_links = []
        for l in pos_train_links:
            while True:
                x = np.random.choice([v for v in G if G.attr[v] == l[2]])
                y = np.random.choice([v for v in G if v != x and G.attr[v] == l[3]])
                if ((x,y) not in mark_pos_links) and ((y,x) not in mark_pos_links) and \
                        ((x,y) not in mark_neg_links) and ((y,x) not in mark_neg_links):
                    break
            mark_neg_links.update([(x,y)])
            neg_train_links.append([x, y, G.attr[x], G.attr[y], 0])

        train_links = pos_train_links + neg_train_links
        test_links = pos_test_links + neg_test_links

        with open(train_links_file, 'wb') as fout:
            pickle.dump(train_links, fout)
        with open(test_links_file, 'wb') as fout:
            pickle.dump(test_links, fout)

  G.make_consistent()
  return G


def load_matfile(file_, variable_name="network", undirected=True):
  mat_varables = loadmat(file_)
  mat_matrix = mat_varables[variable_name]

  return from_numpy(mat_matrix, undirected)

def _expand(G):
  G_p = {v:[u for u in l] for v,l in G.items()}
  for v, l in G_p.items():
    l_1 = [u for u in l if (G.attr[u] == G.attr[v]) and (u != v)]
    l_2 = [u for u in l if G.attr[u] != G.attr[v]]
    for u_2 in l_2:
      tmp = set(G[u_2])
      tmp.update(l_1)
      G[u_2] = list(tmp)
  G.make_consistent()

# def _is_border_node(G, v):
#   nei = G[v]
#   return np.any(np.array([G.attr[u] for u in nei]) != G.attr[v])
#
# def _compute_random_border_distance(G, v, wl):
#   if _is_border_node(G, v):
#     return 0
#
#   cur = v
#   for i in range(1, wl + 1):
#     cur = np.random.choice(G[cur])
#     if _is_border_node(cur):
#       return i
#
#   return wl + 1
#
#
# def _compute_border_score(G, v, wl):
#   return wl + 2 - np.mean([_compute_random_border_distance(G, v, wl) for _ in range(100)])

def _ramdomwalk_colorfulness(G, v, l):
  v_color = G.attr[v]
  cur = v
  res = 0
  for i in range(l):
    cur = np.random.choice(G[cur])
    if G.attr[cur] != v_color:
      res += 1
  return res / l

# cnt_clrf = 0

def  _node_colorfulness(G, v, l):
  # global cnt_clrf
  # if np.mod(cnt_clrf, 100) == 0:
  #   print('cnt_clrf:', cnt_clrf)
  # cnt_clrf += 1
  res = 0.001 + np.mean([_ramdomwalk_colorfulness(G, v, l) for _ in range(1000)])
  return (v, res)

def _colorfulness(G, l):
  # cfn = dict()
  # for i, v in enumerate(G):
  #   print(i, ':')
  #   cfn[v] = _node_colorfulness(G, v)
  # return cfn

  # pool = multiprocessing.Pool(multiprocessing.cpu_count())
  # map_results = pool.starmap(_node_colorfulness, [(G, v) for v in G])
  map_results = [_node_colorfulness(G, v, l) for v in G]
  # pool.close()
  cfn = {k: v for k, v in map_results}
  # print(cfn)
  # asdfkjh
  return cfn

def _set_colored_border_distnaces(G, color):
  queue = [v for v in G if G.attr[v] == color]
  head = 0
  dis = {v:0 for v in queue}
  while head < len(queue):
    cur = queue[head]
    d_cur = dis[cur]
    for u in G[cur]:
      if (G.attr[u] != color) and (u not in dis):
        queue.append(u)
        dis[u] = d_cur + 1
        G.border_distance[u] = d_cur + 1
    head += 1

def _set_border_distances(G):
  G.border_distance = dict()
  _set_colored_border_distnaces(G, 0)
  _set_colored_border_distnaces(G, 1)
  for v in G:
    if v not in G.border_distance:
      G.border_distance[v] = np.inf
  return G

def set_weights(G, method_):
  if method_ is None:
    return G

  if method_.startswith('get_stat'):
    cnt_rb = cnt_br = cnt_b = cnt_r = 0
    for v in G.keys():
      nei = np.array([G.attr[u] for u in G[v]])
      if nei.size == 0:
        raise Exception('Solitary node:', v)
      if np.all( nei == G.attr[v] ):
        if G.attr[v] == 0:
          cnt_b += 1
        elif G.attr[v] == 1:
          cnt_r += 1
        else:
          raise Exception('Bad attr value:', v, G.attr[v])
      else:
        if G.attr[v] == 0:
          cnt_br += 1
        elif G.attr[v] == 1:
          cnt_rb += 1
        else:
          raise Exception('Bad attr value:', v, G.attr[v])
    print('cnt_r=', cnt_r)
    print('cnt_b=', cnt_b)
    print('cnt_rb=', cnt_rb)
    print('cnt_br=', cnt_br)
    khkjhkhkjhkjhk

  if method_.startswith('expandar_'):
    _expand(G)
    method_ = method_[9:]

  if method_ == 'random':
    G.edge_weights = method_
    return G

  if method_.startswith('smartshortcut_'):
    G.edge_weights = method_
    return G

  if method_.startswith('prb_'):
    G.edge_weights = method_
    # tmp = method_.split('_')
    # if len(len(tmp) > 5) and tmp[4] == 'wl':
    #   wl = int(tmp[5])
    #   G.border_score = dict()
    #   for v in G.keys():
    #     G.border_score[v] = _compute_border_score(G, v, wl)
    return G


  if method_.startswith('fairwalk'):
    G.edge_weights = dict()

    for v in G:
      nei_colors = np.unique([G.attr[u] for u in G[v]])
      if nei_colors.size == 0:
        continue
      G.edge_weights[v] = [None for _ in G[v]]
      for cl in nei_colors:
        ind_cl = [i for i, u in enumerate(G[v]) if G.attr[u] == cl]
        sm_cl = len(ind_cl)
        for i in ind_cl:
          G.edge_weights[v][i] = 1 / (nei_colors.size * sm_cl)

    #for v in G:
    #  nei_colors = [G.attr[u] for u in G[v]]
    #  print('\n', G.attr[v])
    #  print(nei_colors)
    #  print(G.edge_weights[v])
    #lkjqlwekrjqew

    return G
 
  if method_.startswith('random_walk'):
    s_method = method_.split('_')
    l = int(s_method[2])
    assert( (s_method[3] in ['bndry', 'revbndry']) and (s_method[5] == 'exp'))
    p_bndry = float(s_method[4])
    exp_ = float(s_method[6])
    cfn = _colorfulness(G, l)
    G.edge_weights = dict()

    for v in G:
      nei_colors = np.unique([G.attr[u] for u in G[v]])
      if nei_colors.size == 0:
        continue
      w_n = [cfn[u] ** exp_ for u in G[v]]
      if nei_colors.size == 1 and nei_colors[0] == G.attr[v]:
        sm = sum(w_n)
        G.edge_weights[v] = [w / sm for w in w_n]
        continue
      G.edge_weights[v] = [None for _ in w_n]
      for cl in nei_colors:
        ind_cl = [i for i, u in enumerate(G[v]) if G.attr[u] == cl]
        w_n_cl = [w_n[i] for i in ind_cl]
        sm_cl = sum(w_n_cl)
        if cl == G.attr[v]:
          coef = (1 - p_bndry)
        else:
          if G.attr[v] in nei_colors:
            coef = p_bndry / (nei_colors.size - 1)
          else:
            coef = 1 / nei_colors.size
        if (s_method[3] == 'bndry'):
          for i in ind_cl:
            G.edge_weights[v][i] = coef * w_n[i] / sm_cl
        else:
          for i in ind_cl:
            G.edge_weights[v][i] = coef * (1 - (w_n[i] / sm_cl)) / (len(ind_cl) - 1)

    '''
    for v in G:
      nei_colors = [G.attr[u] for u in G[v]]
      w_n = [cfn[u] ** exp_ for u in G[v]]
      print('\n', G.attr[v])
      print(nei_colors)
      print(w_n)
      print(G.edge_weights[v])
    '''

    '''
    for v in G:
      w_n = [cfn[u] ** exp_ for u in G[v]]
      ind_same = [i for i, u in enumerate(G[v]) if G.attr[u] == G.attr[v]]
      ind_diff = [i for i, u in enumerate(G[v]) if G.attr[u] != G.attr[v]]
      if len(ind_same) == 0 or len(ind_diff) == 0:
        sm = sum(w_n)
        G.edge_weights[v] = [w/sm for w in w_n]
      else:
        w_n_same = [w_n[i] for i in ind_same]
        w_n_diff = [w_n[i] for i in ind_diff]
        sm_same = sum(w_n_same)
        sm_diff = sum(w_n_diff)
        G.edge_weights[v] = [None for _ in w_n]
        for i in ind_same:
          G.edge_weights[v][i] = (1-p_bndry) * w_n[i] / sm_same
        if (s_method[3] == 'bndry') or (len(ind_diff) == 1):
          for i in ind_diff:
            G.edge_weights[v][i] = p_bndry * w_n[i] / sm_diff
        else:
          l_diff = len(ind_diff)
          for i in ind_diff:
            G.edge_weights[v][i] = p_bndry * (1 - (w_n[i] / sm_diff)) / (l_diff - 1)
    '''

    return G

  if method_.startswith('pch_'):
    G.edge_weights = method_
    G = _set_border_distances(G)
    for c, c_i in [('blue', 0), ('red', 1)]:
      print(c + ' Nodes:')
      l = 1
      while True:
        t = len([v for v in G if ((G.attr[v] == c_i) and (G.border_distance[v] == l))])
        if t == 0:
          break
        print('     level ' + str(l) + ':', t)
        l += 1
      t = np.sum(np.isinf([G.border_distance[v] for v in G if G.attr[v] == c_i]))
      if t > 0:
        print('     level inf:', t)
    # print([d for v,d in G.border_distance.items() if G.attr[v] == 1])
    return G

  if method_.startswith('constant_'):
    c = float(method_[9:])
    G.edge_weights = dict()
    for v in G.keys():
      tmp = [1 if G.attr[u] == G.attr[v] else c for u in G[v]]
      sm = sum(tmp)
      G.edge_weights[v] = [w/sm for w in tmp]
  elif method_.startswith('rb_'):
    s_ = method_.split('_')
    c_rb, c_br = float(s_[1]), float(s_[3])
    G.edge_weights = dict()
    for v in G.keys():
      c = c_rb if G.attr[v] == 1 else c_br
      tmp = [1. if G.attr[u] == G.attr[v] else c for u in G[v]]
      sm = sum(tmp)
      G.edge_weights[v] = [w/sm for w in tmp]
  else:
    raise Exception('Weighting method "' + str(method_) + '" not supported.')
  return G


def from_networkx(G_input, undirected=True):
    G = Graph()

    for idx, x in enumerate(G_input.nodes()):
        for y in iterkeys(G_input[x]):
            G[x].append(y)

    if undirected:
        G.make_undirected()

    return G


def from_numpy(x, undirected=True):
    G = Graph()

    if issparse(x):
        cx = x.tocoo()
        for i,j,v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
      raise Exception("Dense matrices not yet supported.")

    if undirected:
        G.make_undirected()

    G.make_consistent()
    return G


def from_adjlist(adjlist):
    G = Graph()
    
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = list(sorted(set(neighbors)))

    return G


def from_adjlist_unchecked(adjlist):
    G = Graph()
    
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = neighbors

    return G


def compute_heuristic_wrb(G, w_br):
  r_d_r, r_d_b, b_d_r, b_d_b = [], [], [], []
  for v in G:
    d_r = np.sum([(G.attr[u] == 1) for u in G[v]])
    d_b = len(G[v]) - d_r
    if G.attr[v] == 1:
      r_d_r.append(d_r)
      r_d_b.append(d_b)
    else:
      b_d_r.append(d_r)
      b_d_b.append(d_b)
  n_b = np.sum([(G.attr[v] == 0) for v in G])

  def one_step_E(w_rb):
    return sum([d_b / (d_r * w_br + d_b) for d_r, d_b in zip(b_d_r, b_d_b)]) + \
           sum([(w_rb * d_b) / (d_r + w_rb * d_b) for d_r, d_b in zip(r_d_r, r_d_b)])

  L = R = 1.

  while one_step_E(R) < n_b:
    L = R
    R *= 2

  while one_step_E(L) > n_b:
    R = L
    L /= 2

  while True:
    w_rb = (L + R) / 2
    err = one_step_E(w_rb) - n_b
    if abs(err) < 1e-7:
      break
    if err > 0:
      R = w_rb
    else:
      L = w_rb

  return w_rb, err
