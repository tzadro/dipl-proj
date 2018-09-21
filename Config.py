class Config:
	def __init__(self):
		self.connection_mutation_probability = 0.8
		self.perturbation_probability = 0.9
		self.new_node_probability = 0.04
		self.new_connection_probability = 0.08
		self.step = 0.25
		self.innovation_number = 8
		self.node_key = 6
		self.c1 = 1.0
		self.c2 = 1.0
		self.c3 = 0.8
		self.compatibility_threshold = 1.0
		self.crossover_probability = 0.75
		self.disable_probability = 0.75
		self.pop_size = 30
		self.num_iter = 70
		self.survival_threshold = 0.8
		self.new_mu = 0
		self.new_sigma = 1
		self.step_mu = 0
		self.step_sigma = 1
		self.visualize_networks = False
		self.fixed_topology = True

		self.input_keys = None
		self.output_keys = None
		self.action_space_discrete: None
		self.action_space_high: None
		self.action_space_low: None

	def update(self, observation_space, action_space):
		self.action_space_discrete = type(action_space).__name__ == 'Discrete'

		num_inputs = observation_space.shape[0]
		if self.action_space_discrete:
			num_outputs = action_space.n
		else:
			num_outputs = action_space.shape[0]
			self.action_space_high = action_space.high
			self.action_space_low = action_space.low

		self.input_keys = list(range(num_inputs))
		self.output_keys = list(range(num_inputs, num_inputs + num_outputs))


config = Config()
