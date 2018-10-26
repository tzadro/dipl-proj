class Config:
	def __init__(self):
		self.connection_mutation_probability = 0.5
		self.perturbation_probability = 0.9
		self.new_node_probability = 0.003
		self.new_connection_probability = 0.005
		self.step = 0.25
		self.c1 = 1.0
		self.c2 = 1.0
		self.c3 = 0.4
		self.compatibility_threshold = 1.8
		self.crossover_probability = 0.75
		self.stay_disabled_probability = 0.75
		self.pop_size = 100
		self.num_iter = 101
		self.survival_threshold = 0.8
		self.new_mu = 0
		self.new_sigma = 1.0
		self.step_mu = 0
		self.step_sigma = 0.75
		self.min_num_individuals_for_elitism = 5
		self.max_num_generations_before_improvement = 15
		self.num_evaluation_runs = 3

		self.num_starting_hidden_nodes = 8
		self.fixed_topology = True

		self.visualize_best_networks = True
		self.visualize_every = 20
		self.verbose = True
		self.network_canvas_height = 100
		self.network_canvas_width = 100

		self.next_node_key = None
		self.starting_num_connections = None
		self.innovation_number = None
		self.input_keys = None
		self.output_keys = None
		self.action_space_discrete = None
		self.action_space_high = None
		self.action_space_low = None

	def update(self, num_inputs, num_outputs, action_space_discrete, action_space_high, action_space_low):
		self.next_node_key = num_inputs + num_outputs + self.num_starting_hidden_nodes
		self.action_space_discrete = action_space_discrete
		self.action_space_high = action_space_high
		self.action_space_low = action_space_low
		self.starting_num_connections = (self.num_starting_hidden_nodes * (num_inputs + num_outputs)) if self.num_starting_hidden_nodes else (num_inputs * num_outputs)
		self.innovation_number = self.starting_num_connections
		self.input_keys = list(range(num_inputs))
		self.output_keys = list(range(num_inputs, num_inputs + num_outputs))


config = Config()
