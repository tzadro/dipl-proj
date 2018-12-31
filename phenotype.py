from neuron import Neuron
import utility


class Phenotype:
	def __init__(self, connections, nodes, config):
		self.input_keys = config.input_keys
		self.output_keys = config.output_keys
		self.neurons = {}

		for node in nodes:
			self.neurons[node.key] = Neuron(node.key, node.bias)

		for connection in connections:
			if not connection.enabled:
				continue

			self.neurons[connection.from_key].add_outgoing(connection.to_key)

			recurrent = utility.check_if_path_exists_by_neurons(connection.to_key, connection.from_key, self.neurons)
			self.neurons[connection.to_key].add_incoming(connection, recurrent)

	def forward(self, inputs):
		for neuron in self.neurons.values():
			neuron.reset()

		for key, value in zip(self.input_keys, inputs):
			self.neurons[key].set_value(value, self.neurons)

		output = [self.neurons[key].value for key in self.output_keys]
		return output

	def flush(self):
		for neuron in self.neurons.values():
			neuron.value = 0
