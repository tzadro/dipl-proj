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

			from_neuron = self.neurons[connection.from_key]
			from_neuron.outgoing_keys.append(connection.to_key)

			if connection.to_key not in self.neurons:
				self.neurons[connection.to_key] = Neuron(connection.to_key)

			to_neuron = self.neurons[connection.to_key]
			to_neuron.incoming_connections.append(connection)
			if not helperfunctions.check_if_path_exists2(connection.to_key, connection.from_key, self.neurons, {}):
				to_neuron.num_accepts_before_firing = to_neuron.num_accepts_before_firing + 1

	def forward(self, inputs):
		for neuron in self.neurons.values():
			neuron.reset()

		for key, value in zip(config.input_keys, inputs):
			self.neurons[key].set_value(value, self.neurons)

		# todo: ugly
		max_key = None
		max_value = -1
		for key in config.output_keys:
			if self.neurons[key].value > max_value:
				max_key = key
				max_value = self.neurons[key].value

		return max_key
