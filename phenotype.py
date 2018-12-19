from config import config
from neuron import Neuron
import utility


class Phenotype:
	def __init__(self, connections):
		self.neurons = {}

		for key in config.input_keys + config.output_keys:
			self.neurons[key] = Neuron(key)

		for connection in connections:
			if not connection.enabled:
				continue

			if connection.from_key not in self.neurons:
				self.neurons[connection.from_key] = Neuron(connection.from_key)

			self.neurons[connection.from_key].add_outgoing(connection.to_key)

			if connection.to_key not in self.neurons:
				self.neurons[connection.to_key] = Neuron(connection.to_key)

			recurrent = utility.check_if_path_exists_by_neurons(connection.to_key, connection.from_key, self.neurons)
			self.neurons[connection.to_key].add_incoming(connection, recurrent)

	def forward(self, inputs):
		for neuron in self.neurons.values():
			neuron.reset()

		for key, value in zip(config.input_keys, inputs):
			self.neurons[key].set_value(value, self.neurons)

		output = [self.neurons[key].value for key in config.output_keys]
		return output

	def flush(self):
		for neuron in self.neurons.values():
			neuron.value = 0
