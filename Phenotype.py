from Config import config
from Neuron import Neuron
import helperfunctions


class Phenotype:  # Neural network
	def __init__(self, connections):
		self.neurons = {}

		for connection in connections.values():
			if not connection.enabled:
				continue

			if connection.from_key not in self.neurons:
				self.neurons[connection.from_key] = Neuron(connection.from_key)

			self.neurons[connection.from_key].add_outgoing(connection.to_key)

			if connection.to_key not in self.neurons:
				self.neurons[connection.to_key] = Neuron(connection.to_key)

			return_path_exists = helperfunctions.check_if_path_exists2(connection.to_key, connection.from_key, self.neurons)
			self.neurons[connection.to_key].add_incoming(connection, return_path_exists)

	def forward(self, inputs):
		for neuron in self.neurons.values():
			neuron.reset()

		for key, value in zip(config.input_keys, inputs):
			self.neurons[key].set_value(value, self.neurons)

		if not config.action_space_discrete:
			output = [self.neurons[key].value for key in config.output_keys]
			return output * (abs(config.action_space_high) + abs(config.action_space_low)) + config.action_space_low

		max_key = None
		max_value = -1
		for key in config.output_keys:
			if self.neurons[key].value > max_value:
				max_key = key
				max_value = self.neurons[key].value

		return config.output_keys.index(max_key)
