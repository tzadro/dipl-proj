class Config:
	def __init__(self):
		self.connection_mutation_probability = 0.8
		self.perturbation_probability = 0.9
		self.new_node_probability = 0.04
		self.new_connection_probability = 0.08
		self.step = 0.25
		self.c1 = 1.0
		self.c2 = 1.0
		self.c3 = 0.4
		self.compatibility_threshold = 3.0
		self.crossover_probability = 0.75
		self.disable_probability = 0.75
		self.pop_size = 50
		self.num_iter = 100
		self.survival_threshold = 0.8
		self.new_mu = 0
		self.new_sigma = 1
		self.step_mu = 0
		self.step_sigma = 1
		self.min_num_individuals_for_elitism = 5
		self.max_num_generations_before_improvement = 15

		self.visualize_best_networks = True
		self.fixed_topology = True
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
		self.next_node_key = num_inputs + num_outputs
		self.action_space_discrete = action_space_discrete
		self.action_space_high = action_space_high
		self.action_space_low = action_space_low
		self.starting_num_connections = num_inputs * num_outputs
		self.innovation_number = self.starting_num_connections
		self.input_keys = list(range(num_inputs))
		self.output_keys = list(range(num_inputs, num_inputs + num_outputs))


config = Config()
