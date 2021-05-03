import numpy as np

class infMaxConfig(object):
	def __init__(self, args):
		self.synthetic1   =  False
		self.timing_test  =  False
		self.twitter      =  False
		self.facebook     =  False
		self.rice         =  False
		self.rice_subset  =  True
		self.sample_1000  =  False
		self.sample_4000_connected_subset  =  False
		self.synthetic    =  False
		self.synthetic_3g =  False
		self.synthetic_3layers =  False

		if self.synthetic1:

			self.num_nodes = 500
			self.p_with = .025

			self.p_acrosses = [ 0.001, 0.025, 0.015, 0.005] # experiments for dataset params 

			self.p_across =.001 

			self.group_ratios = [0.5,0.55, 0.6, 0.65,0.7]  # experiments for dataset params 

			self.group_ratio = 0.7

			self.gammas_log = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5]

			self.gamma_log = 1.0

			self.gammas_root = [2.0,3.0,4.0,5.0,6.0,8.0,10.,15,30]

			self.gammas_root_majority = [1.1, 1.2, 1.5, 2.0, 2.5, 3.5, 4.5, 5.5, 10.0]

			self.beta_root = [1.0]#,2.0,3.0, 4.0 ]

			self.gamma_root = 2.0

			self.seed_size = 30

			self.types = [1,2]

			self.type = 2

			self.filename = 'results/synthetic_data'

			self.reach_list = [0.2] #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

			self.gamma_timings_a_list = [1.0,0.9,0.8,0.7,0.6]

			self.gamma_timings_b_list = [0.0]

		elif self.timing_test:

			self.filename = 'results/timing_test'
			self.num_nodes = 10 
			self.p_edges = 0.5
			self.weight = .2 
			self.gamma_a = 0.5

		elif self.twitter:

			self.weight = 0.01 # 0.1 
			self.filename = 'twitter/twitter_combined_communities'

		elif self.facebook:

			self.weight = 0.3
			self.filename = 'facebook/facebook_combined'

		elif self.rice:

			self.weight = 0.01
			self.filename = 'facebook/rice'

		elif self.rice_subset:

			self.weight = 0.01
			self.filename = 'facebook/rice_subset'
			self.weighted_graph_filename = 'weighted_graphs/rice_subset'

		elif self.sample_1000:

			self.weight = 0.4
			self.filename = 'sample/sample_1000'

		elif self.sample_4000_connected_subset:

			self.weight = 0.01
			self.filename = 'sample/sample_4000_connected_subset/sample_4000_connected_subset'

		elif self.synthetic:

			self.weight = args.Pact
			self.filename = 'synthetic/synthetic_n' + str(args.nodes) + '_Pred' + str(args.Pred) + \
                                            '_Phom' + str(args.Phom) + '_Phet' + str(args.Phet)

		elif self.synthetic_3layers:

			self.weight = 0.03
			self.filename = 'synthetic_3layers/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.003'

		elif self.synthetic_3g:

			self.weight = args.Pact
			self.filename = 'synthetic_3g/synthetic_3g_n' + str(args.nodes) + '_Pred' + str(args.Pred) + \
							'_Pblue' + str(args.Pblue) + '_Prr' + str(args.Prr) + '_Pbb' + str(args.Pbb) + \
							'_Pgg' + str(args.Pgg) + '_Prb' + str(args.Prb) + '_Prg' + str(args.Prg) + \
							'_Pbg' + str(args.Pbg)



			
