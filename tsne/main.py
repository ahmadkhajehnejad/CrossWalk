from gensim.models import Word2Vec

model = Word2Vec()

import numpy as np
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
# import networkx as nx

def read_embeddings(emb_file):
    emb = dict()
    with open(emb_file, 'r') as fin:
        for i_l, line in enumerate(fin):
            s = line.split()
            if i_l == 0:
                dim = int(s[1])
                continue
            emb[int(s[0])] = [float(x) for x in s[1:]]
    return emb, dim

def read_sensitive_attr(sens_attr_file, emb):
    sens_attr = dict()
    with open(sens_attr_file, 'r') as fin:
        for line in fin:
            s = line.split()
            id = int(s[0])
            if id in emb:
                sens_attr[id] = int(s[1])
    return sens_attr

if __name__ == '__main__':

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # matplotlib.rcParams['text.usetex'] = True

    run_i = ''

    # emb_file = 'rice_subset/rice_subset.embeddings_unweighted_d32' + str(run_i)
    # res_file = 'results/TSNE_rice_subset.unweighted.pdf'
    # emb_file = 'rice_subset/rice_subset.embeddings_random_walk_5_bndry_0.5_exp_2.0_d32' + str(run_i)
    # res_file = 'results/TSNE_rice_subset.random_walk_5_bndry_0.5_exp_2.0.pdf'
    # emb_file = 'rice_subset/rice_subset.embeddings_random_walk_5_bndry_0.5_exp_4.0_d32' + str(run_i)
    # res_file = 'results/TSNE_rice_subset.random_walk_5_bndry_0.5_exp_4.0.pdf'
    # sens_attr_file = 'rice_subset/rice_sensitive_attr.txt'
    #
    # emb_file = 'synthetic/synthetic_n500_Pred0.7_Phom0.025_Phet0.001.embeddings_unweighted_d32' + str(run_i)
    # res_file = 'results/synthetic_n500_Pred0.7_Phom0.025_Phet0.001.unweighted.pdf'
    # emb_file = 'synthetic/synthetic_n500_Pred0.7_Phom0.025_Phet0.001.embeddings_random_walk_5_bndry_0.4_exp_2.0_d32' + str(
    #     run_i)
    # res_file = 'results/synthetic_n500_Pred0.7_Phom0.025_Phet0.001.random_walk_5_bndry_0.4_exp_2.0.pdf'
    # sens_attr_file = 'synthetic/synthetic_n500_Pred0.7_Phom0.025_Phet0.001.attr'

    emb_file = 'synthetic_3layers/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.003.embeddings_unweighted_d32' + str(run_i)
    res_file = 'results/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.003.unweighted.pdf'
    # emb_file = 'synthetic_3layers/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.003.embeddings_random_walk_5_bndry_0.5_exp_2.0_d32' + str(
    #     run_i)
    # res_file = 'results/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.003.random_walk_5_bndry_0.5_exp_2.0.pdf'
    # emb_file = 'synthetic_3layers/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.003.embeddings_random_walk_5_bndry_0.5_exp_4.0_d32' + str(
    #     run_i)
    # res_file = 'results/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.003.random_walk_5_bndry_0.5_exp_4.0.pdf'
    sens_attr_file = 'synthetic_3layers/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.01.attr'


    emb, dim = read_embeddings(emb_file)
    sens_attr = read_sensitive_attr(sens_attr_file, emb)

    assert len(emb) == len(sens_attr)

    n = len(emb)

    X = np.zeros([n, dim])
    z = np.zeros([n])
    for i, id in enumerate(emb):
        X[i,:] = np.array(emb[id])
        z[i] = sens_attr[id]

    X_emb = TSNE().fit_transform(X)

    print(X_emb.shape)

    # G = nx.Graph()

    X_red = X[z == 1, :]
    X_blue = X[z == 0, :]
    X_emb_red = X_emb[z == 1, :]
    X_emb_blue = X_emb[z == 0, :]
    plt.scatter(X_emb_red[:,0], X_emb_red[:,1], color='r', s=5)
    plt.scatter(X_emb_blue[:,0], X_emb_blue[:,1], color='b', s=5)

    plt.savefig(res_file, bbox_inches='tight')


