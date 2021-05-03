#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
from io import open
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging

import graph
import walks as serialized_walks
from gensim.models import Word2Vec
from skipgram import Skipgram

from six import text_type as unicode
from six import iteritems
from six.moves import range

import psutil
from multiprocessing import cpu_count

p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def debug(type_, value, tb):
  if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    sys.__excepthook__(type_, value, tb)
  else:
    import traceback
    import pdb
    traceback.print_exception(type_, value, tb)
    print(u"\n")
    pdb.pm()


def process(args):
    if args.format == "adjlist":
        G = graph.load_adjacencylist(args.input, undirected=args.undirected)
    elif args.format == "edgelist":
        G = graph.load_edgelist(args.input, undirected=args.undirected, attr_file_name=args.sensitive_attr_file, 
                test_links_ratio=args.test_links, test_links_file=args.test_links_file,
                train_links_file=args.train_links_file)
    elif args.format == "mat":
        G = graph.load_matfile(args.input, variable_name=args.matfile_variable_name, undirected=args.undirected)
    else:
        raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % args.format)

    if args.heuristic_wrb_for_wbr is not None:
        wrb, err = graph.compute_heuristic_wrb(G, float(args.heuristic_wrb_for_wbr))
        print(wrb, err)
        return


    if (args.weighted is not None) and (args.weighted != 'unweighted'):
      G = graph.set_weights(G, args.weighted)

    if args.just_write_graph:
        with open('wgraph.out', 'w') as fout:
            if args.weighted == 'unweighted':
                for v in G:
                    s = len(G[v])
                    for u in G[v]:
                        fout.write(str(v) + ' ' + str(u) + ' ' + str(1/s) + '\n')
            elif args.weighted.startswith('random_walk'):
                for v in G:
                    for u, w in zip(G[v], G.edge_weights[v]):
                        fout.write(str(v) + ' ' + str(u) + ' ' + str(w) + '\n')
            else:
                raise Exception('just-write-graph is not supported for this weighting method')
        return None




    num_walks = len(G.nodes()) * args.number_walks

    print("Number of walks: {}".format(num_walks))

    data_size = num_walks * args.walk_length

    print("Data size (walks*length): {}".format(data_size))

    if data_size < args.max_memory_data_size:
        print("Walking...")
        walks = graph.build_deepwalk_corpus(G, num_paths=args.number_walks,
                                            path_length=args.walk_length, p_modified=args.pmodified,
                                            alpha=0, rand=random.Random(args.seed))
        print("Training...")
        model = Word2Vec(walks, size=args.representation_size, window=args.window_size, min_count=0, sg=1, hs=1, workers=args.workers)
    else:
        print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, args.max_memory_data_size))
        print("Walking...")

        walks_filebase = args.output + ".walks"
        walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=args.number_walks,
                                             path_length=args.walk_length, p_modified=args.pmodified,
                                             alpha=0, rand=random.Random(args.seed),
                                             num_workers=args.workers)

        print("Counting vertex frequency...")
        if not args.vertex_freq_degree:
          vertex_counts = serialized_walks.count_textfiles(walk_files, args.workers)
        else:
          # use degree distribution for frequency in tree
          vertex_counts = G.degree(nodes=G.iterkeys())

        print("Training...")
        walks_corpus = serialized_walks.WalksCorpus(walk_files)
        model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                         size=args.representation_size,
                         window=args.window_size, min_count=0, trim_rule=None, workers=args.workers)

    model.wv.save_word2vec_format(args.output)


def main():
  parser = ArgumentParser("deepwalk",
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')

  parser.add_argument("--debug", dest="debug", action='store_true', default=False,
                      help="drop a debugger if an exception is raised.")

  parser.add_argument('--format', default='adjlist',
                      help='File format of input file')

  parser.add_argument('--input', nargs='?', required=True,
                      help='Input graph file')

  parser.add_argument("-l", "--log", dest="log", default="INFO",
                      help="log verbosity level")

  parser.add_argument('--matfile-variable-name', default='network',
                      help='variable name of adjacency matrix inside a .mat file.')

  parser.add_argument('--max-memory-data-size', default=1000000000, type=int,
                      help='Size to start dumping walks to disk, instead of keeping them in memory.')

  parser.add_argument('--number-walks', default=10, type=int,
                      help='Number of random walks to start at each node')

  parser.add_argument('--output', required=True,
                      help='Output representation file')

  parser.add_argument('--representation-size', default=64, type=int,
                      help='Number of latent dimensions to learn for each node.')

  parser.add_argument('--seed', default=0, type=int,
                      help='Seed for random walk generator.')

  parser.add_argument('--undirected', default=True, type=bool,
                      help='Treat graph as undirected.')

  parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                      help='Use vertex degree to estimate the frequency of nodes '
                           'in the random walks. This option is faster than '
                           'calculating the vocabulary.')

  parser.add_argument('--walk-length', default=40, type=int,
                      help='Length of the random walk started at each node')

  parser.add_argument('--window-size', default=5, type=int,
                      help='Window size of skipgram model.')

  parser.add_argument('--workers', default=1, type=int,
                      help='Number of parallel processes.')

  parser.add_argument('-w', '--weighted', default=None, help='Put weights on edges.')

  parser.add_argument('-s', '--sensitive-attr-file', help='sensitive attribute values file path.')

  parser.add_argument('-h', '--heuristic-wrb-for-wbr', help='If set to a value, that value is considered for w_br ' +
                                                            'and w_rb is computed by a heuristic method and returned')
  parser.add_argument('--pmodified', default=1.0, type=float, help='Probability of using the modified graph')
  parser.add_argument('--just-write-graph',
                      help='Do not run the deepwalk, just run the preprocessing and write the resutled weighted graph in file wgraph.out',
                      action='store_true')
  parser.add_argument('--test-links', type=float, default=0., help='Portion of connections used as test data in link prediction.')
  parser.add_argument('--test-links-file', default=None, help='Name of the file of the test links')
  parser.add_argument('--train-links-file', default=None, help='Name of the file of the train links')

  args = parser.parse_args()
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(format=LOGFORMAT)
  logger.setLevel(numeric_level)

  if args.debug:
   sys.excepthook = debug

  process(args)

if __name__ == "__main__":
  sys.exit(main())
