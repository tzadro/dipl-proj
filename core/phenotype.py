from core.config import config
from core.neuron import Neuron
from core import utility


class Phenotype:
	def __init__(self, connections, nodes):
		self.neurons = {}

		# create a node for every neuron
		for node in nodes:
			self.neurons[node.key] = Neuron(node.key, node.bias)

		# iterate through all connections
		for connection in connections:
			# skip disabled connections
			if not connection.enabled:
				continue

			# save incoming key to outgoing neuron
			self.neurons[connection.from_key].add_outgoing(connection.to_key)

			# save connection to incoming neuron
			recurrent = utility.check_if_path_exists_by_neurons(connection.to_key, connection.from_key, self.neurons)
			self.neurons[connection.to_key].add_incoming(connection, recurrent)

	def forward(self, inputs):
		# first reset all neurons
		for neuron in self.neurons.values():
			neuron.reset()

		# set value for input neurons
		for key, value in zip(config.input_keys, inputs):
			self.neurons[key].set_value(value, self.neurons)

		# neurons will automatically propagate and we can collect values from output nodes
		output = [self.neurons[key].value for key in config.output_keys]
		return output

	def flush(self):
		for neuron in self.neurons.values():
			neuron.value = 0
