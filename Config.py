class Config:
	def __init__(self):
		self.connection_mutation_probability = 0.8
		self.perturbation_probability = 0.9
		self.new_node_probability = 0.04
		self.new_connection_probability = 0.08
		self.step = 0.25
		self.node_key = 6
		self.c1 = 1.0
		self.c2 = 1.0
		self.c3 = 0.8
		self.compatibility_threshold = 1.0
		self.crossover_probability = 0.75
		self.disable_probability = 0.75
		self.pop_size = 50
		self.num_iter = 200
		self.survival_threshold = 0.8
		self.new_mu = 0
		self.new_sigma = 1
		self.step_mu = 0
		self.step_sigma = 1
		self.min_num_individuals_for_elitism = 5
		self.max_num_generations_before_improvement = 15

		self.visualize_networks = True
		self.fixed_topology = False
		self.network_canvas_height = 100
		self.network_canvas_width = 100

		self.starting_num_connections = None
		self.innovation_number = None
		self.input_keys = None
		self.output_keys = None
		self.action_space_discrete = None
		self.action_space_high = None
		self.action_space_low = None

	def update(self, observation_space, action_space):
		self.action_space_discrete = type(action_space).__name__ == 'Discrete'

		num_inputs = observation_space.shape[0]
		if self.action_space_discrete:
			num_outputs = action_space.n
		else:
			num_outputs = action_space.shape[0]
			self.action_space_high = action_space.high
			self.action_space_low = action_space.low

		self.starting_num_connections = num_inputs * num_outputs
		self.innovation_number = self.starting_num_connections
		self.input_keys = list(range(num_inputs))
		self.output_keys = list(range(num_inputs, num_inputs + num_outputs))


config = Config()
