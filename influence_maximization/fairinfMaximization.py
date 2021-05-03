''' File for testing different files in parallel
'''

from config import infMaxConfig
from generalGreedy import *
import utils as ut
from IC import *
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import os

import networkx as nx


def dfs(v, mark, G, colors, num_labels):
    res = np.zeros(num_labels)
    res[int(colors[v])] += 1
    mark.update([v])
    for u in G[v]:
        if u not in mark:
            res += dfs(u, mark, G, colors, num_labels)
    return res


class fairInfMaximization(infMaxConfig):

    def __init__(self, num=-1, args=None):
        super(fairInfMaximization, self).__init__(args)

        if self.rice or self.rice_subset or self.sample_1000 or self.sample_4000_connected_subset or self.synthetic or self.synthetic_3g or self.synthetic_3layers:
            self.G = ut.get_data(self.filename, self.weight)

            '''
            colors = nx.get_node_attributes(self.G, 'color')
            n_v = [782, 2598, 180]
            n_e = np.zeros([3,3])
            for e in self.G.edges:
                n_e[int(colors[e[0]]), int(colors[e[1]])] += 1

            print('00', n_e[0,0] // 2, "\t{:.4f}".format(n_e[0,0] / (n_v[0] * (n_v[0]-1))) )
            print('01', n_e[0,1], "\t{:.4f}".format( n_e[0,1] / (n_v[0] * n_v[1])))
            print('02', n_e[0,2], "\t{:.4f}".format( n_e[0,2] / (n_v[0] * n_v[2])))
            print('11', n_e[1,1] // 2, "\t{:.4f}".format(n_e[1,1] / (n_v[1] * (n_v[1]-1))))
            print('12', n_e[1,2], "\t{:.4f}".format( n_e[1,2] / (n_v[1] * n_v[2])))
            print('22', n_e[2,2] // 2, "\t{:.4f}".format( n_e[2,2] / (n_v[2] * (n_v[2]-1))))
            print('total', np.sum(n_e) / 2, "\t{:.4f}".format( np.sum(n_e) / (np.sum(n_v) * (np.sum(n_v)-1))))
            print()
            masldkjfkj
            '''

            '''
            colors = nx.get_node_attributes(self.G, 'color')
            mark = set()
            num_labels = 3
            all_res = []
            for v in self.G.nodes:
                if v not in mark:
                    tmp_mark = set()
                    res = dfs(v, tmp_mark, self.G, colors, num_labels)
                    mark.update(list(tmp_mark))
                    if len(tmp_mark) > 100:
                        break

            with open('sample/sample_4000_connected/sample_4000_connected.attr', 'w') as fout:
                for v in tmp_mark:
                    fout.write(str(v) + ' ' + str(colors[v]) + '\n')
            with open('sample/sample_4000_connected/sample_4000_connected.links', 'w') as fout:
                for e in self.G.edges:
                    if  int(e[0]) < int(e[1]) and (e[0] in tmp_mark) and (e[1] in tmp_mark):
                        fout.write(str(e[0]) + ' ' + str(e[1]) + '\n')
            masldkjfkj
            '''

        if self.twitter:
            self.G = ut.get_twitter_data(self.filename, self.weight)

        # self.stats = ut.graph_stats(self.G)

    def test_greedy(self, filename, budget, G_greedy=None):
        generalGreedy_node_parallel(filename, self.G, budget=budget, gamma=None, G_greedy=G_greedy)

    def test_kmedoids(self, emb_filename, res_filename, budget):

        print(res_filename)

        # stats = ut.graph_stats(self.G, print_stats=False)
        v, em = ut.load_embeddings(emb_filename, self.G.nodes())

        influenced, influenced_grouped = [], []
        seeds = []
        for k in range(1, budget + 1):
            print('--------', k)
            S = ut.get_kmedoids_centers(em, k, v)

            I, I_grouped = map_fair_IC((self.G, S))
            influenced.append(I)
            influenced_grouped.append(I_grouped)

            S_g = {c:[] for c in np.unique([self.G.nodes[v]['color'] for v in self.G.nodes])}
            for n in S:
                c = self.G.nodes[n]['color']
                S_g[c].append(n)

            seeds.append(S_g)  # id's of the seeds so the influence can be recreated

        ut.write_files(res_filename, influenced, influenced_grouped, seeds)

        # return (influenced, influenced_a, influenced_b, seeds_a, seeds_b)

    # def test_kmedoids(self, emb_filename, res_filename, budget):
    #
    #     print(res_filename)
    #
    #     stats = ut.graph_stats(self.G, print_stats=False)
    #     v, em = ut.load_embeddings(emb_filename, self.G.nodes())
    #
    #     influenced, influenced_a, influenced_b = [], [], []
    #     seeds_a, seeds_b = [], []
    #     for k in range(1, budget+1):
    #         print('--------', k)
    #         S = ut.get_kmedoids_centers(em, k, v)
    #
    #         I, I_a, I_b = map_fair_IC((self.G, S))
    #         influenced.append(I)
    #         influenced_a.append(I_a)
    #         influenced_b.append(I_b)
    #         S_red = []
    #         S_blue = []
    #         for n in S:
    #             if self.G.nodes[n]['color'] == 'red':
    #                 S_red.append(n)
    #             else:
    #                 S_blue.append(n)
    #
    #         seeds_a.append(S_red)  # id's of the seeds so the influence can be recreated
    #         seeds_b.append(S_blue)
    #
    #     ut.plot_influence(influenced_a, influenced_b, budget, res_filename, stats['group_a'], stats['group_b'],
    #                       [len(S_a) for S_a in seeds_a], [len(S_b) for S_b in seeds_b])
    #
    #     ut.write_files(res_filename, influenced, influenced_a, influenced_b, seeds_a, seeds_b)
    #
    #     # return (influenced, influenced_a, influenced_b, seeds_a, seeds_b)

if __name__ == '__main__':

    parser = ArgumentParser("Synthesize Graph",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--nodes', type=int, help='Number nodes')
    parser.add_argument('--Pred', type=float, help='Probability of being red for each node')
    parser.add_argument('--Phom', type=float, help='Probability of within group connections')
    parser.add_argument('--Phet', type=float, help='Probability of cross group connections')
    parser.add_argument('--Pact', type=float, help='Activation probability of each edge')
    parser.add_argument('--Pblue', type=float, help='Probability of being blue for each node')
    parser.add_argument('--Prr', type=float, help='Probability of within group connections for red nodes')
    parser.add_argument('--Pbb', type=float, help='Probability of within group connections for blue nodes')
    parser.add_argument('--Pgg', type=float, help='Probability of within group connections for green nodes')
    parser.add_argument('--Prb', type=float, help='Probability of connections between red and blue')
    parser.add_argument('--Prg', type=float, help='Probability of connections between red and green')
    parser.add_argument('--Pbg', type=float, help='Probability of connections between blue and green')

    args = parser.parse_args()


    import time

    start = time.time()

    if False:
        fair_inf = fairInfMaximization(args=args)
        d = 'd32'
        tmp_filename = 'rice_subset'
        embfilename = 'facebook/' + tmp_filename + '.embeddings_'
        resfilename = 'rsults/' + tmp_filename
        
        for rwl in [5]:  # [5, 10, 20]:
            for exponent in ['6.0', '8.0', '4.0', '2.0']:  # [0.5, '1.0', '2.0']:
                for bndry in [0.9, 0.8, 0.7, 0.6, 0.5]:  # [0.2, 0.5, 0.7, 0.9]:
                    for bndry_type in ['bndry']:  # ['bndry', 'revbndry']:
                        method = 'random_walk_' + str(rwl) + '_' + bndry_type + '_' + str(bndry) + '_exp_' + str(
                            exponent)
                        fair_inf.test_kmedoids(embfilename + method + '_' + d, resfilename + '_emb_' + method + '_' + d,
                                               budget=40)

        #fair_inf.test_greedy(resfilename, budget=40)
        method = 'unweighted'
        fair_inf.test_kmedoids(embfilename + method + '_' + d, resfilename + '_emb_' + method + '_' + d, budget=40)


    if False:
        fair_inf = fairInfMaximization(args=args)
        d = 'd32'
        tmp_filename = 'synthetic_3g_n' + str(args.nodes) + '_Pred' + str(args.Pred) + \
                       '_Pblue' + str(args.Pblue) + '_Prr' + str(args.Prr) + '_Pbb' + str(args.Pbb) + \
					   '_Pgg' + str(args.Pgg) + '_Prb' + str(args.Prb) + '_Prg' + str(args.Prg) + \
					   '_Pbg' + str(args.Pbg)
        embfilename = 'synthetic_3g/' + tmp_filename + '.embeddings_'
        resfilename = 'rsults/' + tmp_filename + '_Pact' + str(args.Pact)
        # fair_inf.test_greedy(resfilename, budget=40)
        # for method in ['unweighted', 'constant_100', 'constant_1000', 'prb_0.5_pbr_0.5', 'prb_0.7_pbr_0.7', 'prb_0.9_pbr_0.9', 'pch_0.5', 'pch_0.7', 'pch_0.9', 'random']:
        # for method in ['random_walk_5']:
        #    fair_inf.test_kmedoids(embfilename + method + '_' + d, resfilename + '_emb_' + method + '_' + d, budget=40)

        # method = 'pch_0.9'
        # for pm in [0.2, 0.5, 0.7, 0.9, '1.0']:
        #    embfilename = 'synthetic/' + tmp_filename + '.pmodified_' + str(pm) + '_embeddings_'
        #    resfilename = 'rsults/' + tmp_filename + '_pmodified_' + str(pm) + '_Pact' + str(args.Pact)
        #    fair_inf.test_kmedoids(embfilename + method + '_' + d, resfilename + '_emb_' + method + '_' + d, budget=40)

        
        for i in ['1','2','3','4','5']:
            for rwl in [5]:  # [5, 10, 20]:
                for exponent in ['4.0']: #['6.0', '8.0', '4.0', '2.0']:  # [0.5, '1.0', '2.0']:
                    for bndry in [0.5, 0.3]: # [0.9, 0.8, 0.7, 0.6, 0.5]:  # [0.2, 0.5, 0.7, 0.9]:
                        for bndry_type in ['bndry']:  # ['bndry', 'revbndry']:
                            method = 'random_walk_' + str(rwl) + '_' + bndry_type + '_' + str(bndry) + '_exp_' + str(exponent)
                            # embfilename = 'synthetic_3g/' + tmp_filename + '.embeddings_'
                            # resfilename = 'rsults/' + tmp_filename + '_Pact' + str(args.Pact)
                            print(i, '     ', method)
                            if os.path.isfile(resfilename + '_emb_' + method + '_' + d + '_' + i + '_results.txt'):
                                print('exists')
                                continue
                            fair_inf.test_kmedoids(embfilename + method + '_' + d + '_' + i, resfilename + '_emb_' + method + '_' + d + '_' + i,
                                               budget=40)
        

            #fair_inf.test_greedy(resfilename, budget=40)
            #method = 'unweighted'
            #method = 'fairwalk'
            #print(i, '     ', method)
            #if os.path.isfile(resfilename + '_emb_' + method + '_' + d + '_' + i):
            #    print('exists')
            #fair_inf.test_kmedoids(embfilename + method + '_' + d + '_' + i, resfilename + '_emb_' + method + '_' + d + '_' + i, budget=40)

            embfilename_rnd = 'synthetic_3g/' + tmp_filename + '.'
            method = 'randomembedding'
            fair_inf.test_kmedoids(embfilename_rnd + method + '_' + d + '_' + i, resfilename + '_emb_' + method + '_' + d + '_' + i, budget=40)

    if False:
        fair_inf = fairInfMaximization(args=args)
        d = 'd32'
        tmp_filename = 'synthetic_n' + str(args.nodes) + '_Pred' + str(args.Pred) + \
                    '_Phom' + str(args.Phom) + '_Phet' + str(args.Phet)
        embfilename = 'synthetic/' + tmp_filename + '.embeddings_'
        resfilename = 'rsults/' + tmp_filename + '_Pact' + str(args.Pact)
        #fair_inf.test_greedy(resfilename, budget=40)
        #for method in ['unweighted', 'constant_100', 'constant_1000', 'prb_0.5_pbr_0.5', 'prb_0.7_pbr_0.7', 'prb_0.9_pbr_0.9', 'pch_0.5', 'pch_0.7', 'pch_0.9', 'random']:
        #for method in ['random_walk_5']:
        #    fair_inf.test_kmedoids(embfilename + method + '_' + d, resfilename + '_emb_' + method + '_' + d, budget=40)
    
        #method = 'pch_0.9'
        #for pm in [0.2, 0.5, 0.7, 0.9, '1.0']:
        #    embfilename = 'synthetic/' + tmp_filename + '.pmodified_' + str(pm) + '_embeddings_'
        #    resfilename = 'rsults/' + tmp_filename + '_pmodified_' + str(pm) + '_Pact' + str(args.Pact)
        #    fair_inf.test_kmedoids(embfilename + method + '_' + d, resfilename + '_emb_' + method + '_' + d, budget=40)
    
        for i in ['_1', '_2', '_3', '_4', '_5']:
            for rwl in [5]: #[5, 10, 20]:
                for bndry in [0.7, 0.3]: #[0.5, 0.6, 0.7, 0.8, 0.9]: #[0.2, 0.5, 0.7, 0.9]:
                    for exponent in ['4.0']: #['2.0', '4.0', '6.0', '8.0']: #[0.5, '1.0', '2.0']:
                        for bndry_type in ['bndry']: #['bndry', 'revbndry']:
                            method = 'random_walk_' + str(rwl) + '_' + bndry_type + '_' + str(bndry) + '_exp_' + str(exponent)
                            embfilename = 'synthetic/' + tmp_filename + '.embeddings_'
                            resfilename = 'rsults/' + tmp_filename + '_Pact' + str(args.Pact)
                            if os.path.isfile(resfilename + '_emb_' + method + '_' + d + i + '_results.txt'):
                                continue
                            print('===========   ', i, '    ', method)
                            fair_inf.test_kmedoids(embfilename + method + '_' + d + i, resfilename + '_emb_' + method + '_' + d + i, budget=40)
    
    
            # fair_inf.test_greedy(resfilename, budget=40)
            # method = 'unweighted'
            # fair_inf.test_kmedoids(embfilename + method + '_' + d + i, resfilename + '_emb_' + method + '_' + d + i, budget=40)
            embfilename = 'synthetic/' + tmp_filename + '.'
            method = 'randomembedding'
            fair_inf.test_kmedoids(embfilename + method + '_' + d + i, resfilename + '_emb_' + method + '_' + d + i, budget=40)

    if False:
        fair_inf = fairInfMaximization(args=args)
        tmp_filename = 'synthetic_n' + str(args.nodes) + '_Pred' + str(args.Pred) + \
                    '_Phom' + str(args.Phom) + '_Phet' + str(args.Phet)
        resfilename = 'rsults_R500/' + tmp_filename + '_Pact' + str(args.Pact)
        fair_inf.test_greedy(resfilename, budget=40)

    if False:
        fair_inf = fairInfMaximization(args=args)
        resfilename = 'rsults_R500/sample_4000_0.01'
        fair_inf.test_greedy(resfilename, budget=40)

    if False:
        fair_inf = fairInfMaximization(args=args)
        tmp_filename = 'synthetic_3g_n' + str(args.nodes) + '_Pred' + str(args.Pred) + \
                       '_Pblue' + str(args.Pblue) + '_Prr' + str(args.Prr) + '_Pbb' + str(args.Pbb) + \
					   '_Pgg' + str(args.Pgg) + '_Prb' + str(args.Prb) + '_Prg' + str(args.Prg) + \
					   '_Pbg' + str(args.Pbg)
        resfilename = 'rsults_R500/' + tmp_filename + '_Pact' + str(args.Pact)
        fair_inf.test_greedy(resfilename, budget=40)

    if False:

        base_embfile = 'sample/sample_4000_connected_subset/sample_4000_connected_subset' # 'facebook/rice_subset' # 'synthetic_3layers/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.003' # 
        base_resfile = 'rsults/sample_4000_connected_subset_0.01' # 'rsults/rice_subset' # 'rsults/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.003_Pact0.03' # 

        fair_inf = fairInfMaximization(args=args)
        
        d = 'd32'
        
	
        for i in ['1','2','3','4','5']:
            '''
            method = 'unweighted'
            embfilename = base_embfile + '.embeddings_'
            resfilename = base_resfile 
            print(i, '   ', method)
            if os.path.isfile(resfilename + '_emb_' + method + '_' + d + '_' + i + '_results.txt'):
                print('      exists')
            else:
                fair_inf.test_kmedoids(embfilename + method + '_' + d + '_' + i, resfilename + '_emb_' + method + '_' + d + '_' + i, budget=40)
            '''

            method = 'randomembedding'
            embfilename = base_embfile + '.'
            resfilename = base_resfile
            print(i, '   ', method)
            if os.path.isfile(resfilename + '_emb_' + method + '_' + d + '_' + i + '_results.txt'):
                print('      exists')
            else:
                fair_inf.test_kmedoids(embfilename + method + '_' + d + '_' + i, resfilename + '_emb_' + method + '_' + d + '_' + i, budget=40)

            '''
            for rwl in [5]: # [5, 10, 20]:
                for bndry in [0.5]: #[0.2, 0.5, 0.7, 0.9]:
                    for exponent in ['0.0']: # [0.5, '1.0', '2.0']:
                        for bndry_type in ['bndry']: # ['bndry', 'revbndry']:
                            method = 'random_walk_' + str(rwl) + '_' + bndry_type + '_' + str(bndry) + '_exp_' + str(exponent)
                            embfilename = base_embfile + '.embeddings_'
                            resfilename = base_resfile
                            print(i, '   ', method)

                            if os.path.isfile(resfilename + '_emb_' + method + '_' + d + '_' + i + '_results.txt'):
                                print('      exists')
                                continue

                            fair_inf.test_kmedoids(embfilename + method + '_' + d + '_' + i, resfilename + '_emb_' + method + '_' + d + '_' + i, budget=40)
            '''

    if False:
        fair_inf = fairInfMaximization(args=args)
        
        d = 'd32'
        
        #fair_inf.test_kmedoids('sample/sample_4000.embeddings_unweighted_' + d,
        #                                  'rsults/sample_4000' + '_emb_unweighted_' + d, budget=40)

        #fair_inf.test_kmedoids('facebook/rice.embeddings_' + 'random' + '_' + d,
        #                            'rsults/rice_emb_' + 'random' + '_' + d, budget=40)
        
        #for pm in [0.2, 0.5, 0.7, 0.9, '1.0']:
        #    w = 'pch_0.9'
        #    fair_inf.test_kmedoids('facebook/rice.pmodified_' +  str(pm) + '_embeddings_' + w + '_' + d,
        #                                'rsults/rice_pmodified_' + str(pm)  + '_emb_' + w + '_' + d, budget=40)

        #for p_sc in [0.9, 0.7, 0.5, 0.2]:
        #    w = 'smartshortcut_' + str(p_sc)
        #    fair_inf.test_kmedoids('facebook/rice.embeddings_' + w + '_' + d,
        #                                'rsults/rice_emb_' + w + '_' + d, budget=40)
        #for p_ch in [0.9]: # [0.7, 0.5, 0.2]:
        #    w = 'pch_' + str(p_ch)
        #    fair_inf.test_kmedoids('sample/sample_1000.embeddings_' + w + '_' + d,
        #                                'rsults/sample_1000_0.4_emb_' + w + '_' + d, budget=40)

        #for d in ['d64', 'd92', 'd128']:
        #    for p in [0.7]:
        #        w = 'prb_' + str(p) + '_pbr_' + str(p)
        #        fair_inf.test_kmedoids('sample/sample_1000.embeddings_' + w + '_' + d,
        #                                'rsults/sample_1000_0.4_emb_' + w + '_' + d, budget=40)

        #for w in ['wconstant'+str(c) for c in [10, 100, 1000]]:
        #    #for w in ['wexpandconstant'+str(c) for c in [1000,]]: #[7,10,50,100,1000]]:
        #    fair_inf.test_kmedoids('sample/sample_4000_0.4.embeddings_' + w + '_' + d,
        #                                        'rsults/sample_4000_0.4' + '_emb_' + w + '_' + d, budget=40)

        for i in ['1','2','3','4','5']:
            #method = 'unweighted'
            method = 'fairwalk'
            embfilename = 'sample/sample_4000_connected_subset/sample_4000_connected_subset' + '.embeddings_'
            resfilename = 'rsults/sample_4000_connected_subset_0.01'
            print(i, '   ', method)
            fair_inf.test_kmedoids(embfilename + method + '_' + d + '_' + i, resfilename + '_emb_' + method + '_' + d + '_' + i, budget=40)
        #    for rwl in [5]: # [5, 10, 20]:
        #        for bndry in [0.7]: #[0.2, 0.5, 0.7, 0.9]:
        #            for exponent in ['2.0', '4.0']: # [0.5, '1.0', '2.0']:
        #                for bndry_type in ['bndry']: # ['bndry', 'revbndry']:
        #                    method = 'random_walk_' + str(rwl) + '_' + bndry_type + '_' + str(bndry) + '_exp_' + str(exponent)
        #                    embfilename = 'sample/sample_4000_connected_subset/sample_4000_connected_subset' + '.embeddings_'
        #                    resfilename = 'rsults/sample_4000_connected_subset_0.01'
        #                    print(i, '   ', method)
        #                    fair_inf.test_kmedoids(embfilename + method + '_' + d + '_' + i, resfilename + '_emb_' + method + '_' + d + '_' + i, budget=40)

    if False:
        fair_inf = fairInfMaximization(args=args)
        resfilename = 'rsults_R500/sample_4000_connected_subset_0.01'
        fair_inf.test_greedy(resfilename, budget=40)

    if False:
        fair_inf = fairInfMaximization(args=args)
        for d in ['32', '64', '92', '128']:
            list_filenames = ['rsults/rice_greedy__results.txt'] + \
                             ['rsults/rice_emb_unweighted_d' + d + '_results.txt'] + \
                             ['rsults/rice_emb_wconstant' + s + '_d' + d + '_results.txt' for s in ['50', '100', '1000']] # ['2', '3', '5', '7', '10']]
            for type in ['total', 'diff', 'total_frac', 'diff_frac']: # ['total_frac', 'diff_frac']: #
                fair_inf.compare(list_filenames, 'rsults/compare_2/compare_emb' + d + '_' + type + '.png', type)


    if False:
        fair_inf = fairInfMaximization(args=args)
        fair_inf.test_greedy('rsults/twitter', budget=40)

    if False:
        fair_inf = fairInfMaximization(args=args)
        
        method = 'unweighted'
        print('\n\n\n', method)
        embfilename = 'facebook/rice_subset.lineorder1embeddings_' + method + '_d32'
        resfilename = 'rsults/rice_subset_lineorder1emb_' + method + '_d32'
        fair_inf.test_kmedoids(embfilename, resfilename, budget=40)
        '''
        method = 'simpleweighted'
        print('\n\n\n', method)
        embfilename = 'facebook/rice_subset.lineorder1embeddings_' + method + '_d32'
        resfilename = 'rsults/rice_subset_lineorder1emb_' + method + '_d32'
        fair_inf.test_kmedoids(embfilename, resfilename, budget=40)
        '''
        bndry = '0.8'
        for exp_ in ['2.0', '1.5', '1.0']: # ['8.0','6.0','4.0','2.0','1.5','1.0']:
            method = 'random_walk_5_bndry_' + bndry + '_exp_' + exp_
            print('\n\n\n', method)
            embfilename = 'facebook/rice_subset.lineorder1embeddings_' + method + '_d32'
            resfilename = 'rsults/rice_subset_lineorder1emb_' + method + '_d32'
            fair_inf.test_kmedoids(embfilename, resfilename, budget=40)

    if False:
        fair_inf = fairInfMaximization(args=args)
        
        '''
        method = 'unweighted'
        print('\n\n\n', method)
        embfilename = 'facebook/rice_subset.node2vec-p0.5-q0.5embeddings_' + method + '_d30'
        resfilename = 'rsults/rice_subset_node2vec-p0.5-q0.5emb_' + method + '_d30'
        fair_inf.test_kmedoids(embfilename, resfilename, budget=40)
        '''
        '''
        method = 'simpleweighted'
        print('\n\n\n', method)
        embfilename = 'facebook/rice_subset.lineorder1embeddings_' + method + '_d32'
        resfilename = 'rsults/rice_subset_lineorder1emb_' + method + '_d32'
        fair_inf.test_kmedoids(embfilename, resfilename, budget=40)
        '''
        bndry = '0.5'
        for exp_ in ['4.0']: # ['8.0','6.0','4.0','2.0','1.5','1.0']:
            method = 'random_walk_5_bndry_' + bndry + '_exp_' + exp_
            print('\n\n\n', method)
            embfilename = 'facebook/rice_subset.node2vec-p0.5-q0.5embeddings_' + method + '_d30'
            resfilename = 'rsults/rice_subset_node2vec-p0.5-q0.5emb_' + method + '_d30'
            fair_inf.test_kmedoids(embfilename, resfilename, budget=40)


    # Test weighted greedy
    if False:
        fair_inf = fairInfMaximization(args=args)
        d = 'd32'
        tmp_filename = 'rice_subset'

        type_algo = 2
        for rwl in [5]:  # [5, 10, 20]:
            for bndry in [0.5]: #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                for exponent in ['4.0']: #['1.0', '1.25', '1.5', '1.75', '2.0', '4.0', '6.0', '8.0']:
                    for bndry_type in ['bndry']:  # ['bndry', 'revbndry']:
                        method = 'random_walk_' + str(rwl) + '_' + bndry_type + '_' + str(bndry) + '_exp_' + str(
                            exponent)
                        print(fair_inf.filename, '      ', method)
                        resfilename = 'rsults/' + tmp_filename + '_w' + str(type_algo) + '_' + method
                        if os.path.isfile(resfilename + '_greedy__results.txt'):
                            print('      exists')
                            continue
                        W_G = ut.make_weighted_graph(fair_inf.G, 'weighted_graphs/' + tmp_filename + '_' + method + '.txt', type_algo)
                        '''
                        while True:
                            idx = int(input())
                            u = list(W_G.nodes)[idx]
                            print('\n')
                            print(u)
                            for e in W_G.edges:
                                if e[0] == u:
                                    print(e[0], ' ', e[1], ' ', W_G.edges[e]['weight'], ' ', fair_inf.G.edges[e]['weight'])
                            print('  =================================== \n')
                        '''
                        #fair_inf.test_greedy(resfilename + '_R100', budget=40, G_greedy=W_G)
                        fair_inf.test_greedy(resfilename, budget=40, G_greedy=W_G)

        # fair_inf.test_greedy(resfilename, budget=40)
        # method = 'unweighted'
        # fair_inf.test_kmedoids(embfilename + method + '_' + d, resfilename + '_emb_' + method + '_' + d, budget=40)

    if True:
        fair_inf = fairInfMaximization(args=args)
        
        d = 'd32'
        
        for i in ['1','2','3','4','5']:
            embfilename = 'fairwalknode2vec_embeddings/rice-node2vec_' + i
            resfilename = 'fairwalknode2vec_results/rice-node2vec_' + i
            print(i, ' ----')
            fair_inf.test_kmedoids(embfilename, resfilename, budget=40)

    if False:
        fair_inf = fairInfMaximization(args=args)
        resfilename = 'rsults_R500/sample_4000_connected_subset_0.01'
        fair_inf.test_greedy(resfilename, budget=40)

    if False:
        fair_inf = fairInfMaximization(args=args)
        for d in ['32', '64', '92', '128']:
            list_filenames = ['rsults/rice_greedy__results.txt'] + \
                             ['rsults/rice_emb_unweighted_d' + d + '_results.txt'] + \
                             ['rsults/rice_emb_wconstant' + s + '_d' + d + '_results.txt' for s in ['50', '100', '1000']] # ['2', '3', '5', '7', '10']]
            for type in ['total', 'diff', 'total_frac', 'diff_frac']: # ['total_frac', 'diff_frac']: #
                fair_inf.compare(list_filenames, 'rsults/compare_2/compare_emb' + d + '_' + type + '.png', type)




    print('Total time:', time.time() - start)

