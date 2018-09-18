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
		self.input_keys = [0, 1, 2, 3]
		self.output_keys = [4, 5]
		self.actions = {
			self.output_keys[0]: 0,
			self.output_keys[1]: 1
		}
		self.pop_size = 30
		self.num_iter = 70
		self.num_to_remove = 2
		self.new_mu = 0
		self.new_sigma = 1
		self.step_mu = 0
		self.step_sigma = 1
		self.visualize = True
		self.fixed_topology = True


config = Config()
