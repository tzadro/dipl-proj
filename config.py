class Config:
	def __init__(self):
		self.pop_size = 150
		self.c1 = 1.0
		self.c2 = 1.0
		self.c3 = 0.4
		self.compatibility_threshold = 3.0
		self.max_num_generations_before_species_improvement = 15
		self.max_num_generations_before_population_improvement = 20
		self.min_num_individuals_for_elitism = 5
		self.connection_mutation_probability = 0.8
		self.perturbation_probability = 0.9
		self.stay_disabled_probability = 0.75
		self.skip_crossover_probability = 0.25
		self.new_node_probability = 0.03
		self.new_connection_probability = 0.05
		self.sigmoid_coef = 4.9

		self.num_iter = 101
		self.num_runs = 100
		self.survival_threshold = 0.2
		self.new_mu = 0.0
		self.new_sigma = 1.0
		self.step_mu = 0.0
		self.step_sigma = 1.0
		self.normalize = False
		self.disable_self_loops = True

		# used only for new neat
		self.elitism = 2
		self.species_elitism = 2

		# used only for stanley neat
		self.stagnation_penalization = 0.01
		self.youth_threshold = 10
		self.youth_boost = 1.0
		self.mate_only_probability = 0.2

		self.num_starting_hidden_nodes = 0
		self.fixed_topology = False

		self.network_canvas_height = 100
		self.network_canvas_width = 100
		self.visualize_every = 50
		self.visualize_best_networks = True
		self.verbose = False

		self.num_starting_nodes = None
		self.next_node_key = None
		self.num_starting_connections = None
		self.innovation_number = None
		self.input_keys = None
		self.output_keys = None

	def update(self, num_inputs, num_outputs):
		self.num_starting_nodes = num_inputs + num_outputs + self.num_starting_hidden_nodes
		self.next_node_key = self.num_starting_nodes
		self.num_starting_connections = (self.num_starting_hidden_nodes * (num_inputs + num_outputs)) if self.num_starting_hidden_nodes else (num_inputs * num_outputs)
		self.innovation_number = self.num_starting_connections
		self.input_keys = list(range(num_inputs))
		self.output_keys = list(range(num_inputs, num_inputs + num_outputs))


config = Config()
