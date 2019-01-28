from core import utility


class Neuron:
	def __init__(self, key, bias):
		self.key = key
		self.bias = bias
		self.incoming_connections = []
		self.outgoing_keys = []
		self.num_non_recurrent_incoming = 0
		self.num_accepts_before_triggering = None
		self.value = 0

	def add_incoming(self, connection, recurrent):
		self.incoming_connections.append(connection)

		if not recurrent:
			self.num_non_recurrent_incoming += 1

	def add_outgoing(self, key):
		self.outgoing_keys.append(key)

	def reset(self):
		self.num_accepts_before_triggering = self.num_non_recurrent_incoming

	def accept(self, neurons):
		self.num_accepts_before_triggering -= 1

		# if all incoming connections have triggered calculate value and propagate activation
		if self.num_accepts_before_triggering == 0:
			self.calculate_value(neurons)
			self.trigger_outgoing(neurons)

	def calculate_value(self, neurons):
		# start from bias
		score = self.bias

		# for every incoming connection add incoming nodes value adjusted by connection weight
		for connection in self.incoming_connections:
			from_neuron = neurons[connection.from_key]
			score += connection.weight * from_neuron.value

		# run sum through sigmoid activation
		self.value = utility.sigmoid(score)

	def set_value(self, value, neurons):
		self.value = value
		self.trigger_outgoing(neurons)

	def trigger_outgoing(self, neurons):
		# activate neurons for every outgoing connection
		for key in self.outgoing_keys:
			outgoing_neuron = neurons[key]
			outgoing_neuron.accept(neurons)
