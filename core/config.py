class Config:
	def __init__(self):
		# simulation
		self.num_iter = 101
		self.num_runs = 100

		# population
		self.pop_size = 150
		self.elitism = 2
		self.survival_threshold = 0.2
		self.min_num_species = 2
		self.max_num_generations_before_species_improvement = 20

		# distance
		self.c1 = 1.0
		self.c2 = 1.0
		self.c3 = 0.5
		self.compatibility_threshold = 3.0
		self.normalize = True

		# compatibility_threshold
		self.adjust_compatibility_threshold = True
		self.desired_num_species = 5
		self.ct_step = 0.3
		self.ct_min_val = 1.5
		self.ct_max_val = 4.5

		# mutation
		self.new_node_probability = 0.03
		self.new_connection_probability = 0.05
		self.stay_disabled_probability = 0.75
		self.disable_self_loops = True
		self.survival_threshold = 0.2

		# weight
		self.weight_new_mu = 0.0
		self.weight_new_sigma = 1.0
		self.weight_step_mu = 0.0
		self.weight_step_sigma = 0.5
		self.weight_perturbation_probability = 0.8
		self.weight_replace_probability = 0.1

		# bias
		self.bias_new_mu = 0.0
		self.bias_new_sigma = 1.0
		self.bias_step_mu = 0.0
		self.bias_step_sigma = 0.5
		self.bias_perturbation_probability = 0.7
		self.bias_replace_probability = 0.1

		# other
		self.sigmoid_coef = 4.9
		self.tournament_size = 3

		# interface
		self.verbose = False
		self.visualize_best_networks = True
		self.visualize_every = 20
		self.network_canvas_height = 100
		self.network_canvas_width = 100

		# network
		self.next_node_key = None
		self.innovation_number = None
		self.input_keys = None
		self.output_keys = None

	def update(self, num_inputs, num_outputs):
		self.next_node_key = num_inputs + num_outputs
		self.innovation_number = num_inputs * num_outputs
		self.input_keys = list(range(num_inputs))
		self.output_keys = list(range(num_inputs, num_inputs + num_outputs))


config = Config()
