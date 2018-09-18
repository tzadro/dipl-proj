import helperfunctions


class Neuron:
	def __init__(self, key):
		self.key = key
		self.incoming_connections = []
		self.outgoing_keys = []
		self.num_accepts_before_firing = 0
		self.count = None
		self.value = 0

	def reset(self):
		self.count = self.num_accepts_before_firing

	def accept(self, neurons):
		self.count = self.count - 1

		if self.count == 0:
			score = 0

			for connection in self.incoming_connections:
				from_neuron = neurons[connection.from_key]
				score = score + connection.weight * from_neuron.value

			self.value = helperfunctions.sigmoid(score)

			self.trigger_outgoing(neurons)

	def set_value(self, value, neurons):
		self.value = value
		self.trigger_outgoing(neurons)

	def trigger_outgoing(self, neurons):
		for key in self.outgoing_keys:
			outgoing_neuron = neurons[key]
			outgoing_neuron.accept(neurons)
