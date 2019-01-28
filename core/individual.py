from core.config import config
from core.connection import Connection
from core.node import Node
from core import utility
import copy
import random


class Individual:
	def __init__(self, connections=None, nodes=None):
		self.nodes = {}
		self.connections = {}
		self.fitness = None
		self.adjusted_fitness = None

		if connections and nodes:
			self.connections = connections
			self.nodes = nodes
		else:
			self.configure_new()

	def configure_new(self):
		# create a node for every input and output defined by the problem
		for key in config.input_keys + config.output_keys:
			# pick random bias value with gaussian distribution
			bias = random.gauss(config.bias_new_mu, config.bias_new_sigma)
			node = Node(key, bias)
			self.nodes[key] = node

		next_innovation_number = 0
		# fully connect inputs and outputs, i.e. create a connection between every input and output node
		for input_key in config.input_keys:
			for output_key in config.output_keys:
				# pick random connection weight value with gaussian distribution
				new_connection = Connection(next_innovation_number, input_key, output_key, random.gauss(config.weight_new_mu, config.weight_new_sigma), True)
				self.connections[next_innovation_number] = new_connection
				next_innovation_number += 1

	def mutate(self, generation_new_nodes, generation_new_connections):
		# new connection mutation
		if random.random() < config.new_connection_probability:
			self.new_connection(generation_new_connections)

		# new node mutation
		if random.random() < config.new_node_probability:
			self.new_node(generation_new_nodes, generation_new_connections)

		# connection weight and node bias mutations
		self.mutate_connections()
		self.mutate_nodes()

	def mutate_connections(self):
		# go through all connections and either adjust weight by a small amount or assign a new random one
		for connection in self.connections.values():
			r = random.random()
			if r < config.weight_perturbation_probability:
				connection.weight += random.gauss(config.weight_step_mu, config.weight_step_sigma)
			elif r < config.weight_perturbation_probability + config.weight_replace_probability:
				connection.weight = random.gauss(config.weight_new_mu, config.weight_new_sigma)

	def mutate_nodes(self):
		# go through all nodes and either adjust bias by a small amount or assign a new random one
		for node in self.nodes.values():
			r = random.random()
			if r < config.bias_perturbation_probability:
				node.bias += random.gauss(config.bias_step_mu, config.bias_step_sigma)
			elif r < config.bias_perturbation_probability + config.bias_replace_probability:
				node.bias = random.gauss(config.bias_new_mu, config.bias_new_sigma)

	def new_connection(self, generation_new_connections):
		node_keys = list(self.nodes.keys())
		num_nodes = len(node_keys)
		num_connections = len(self.connections.values())
		num_inputs = len(config.input_keys)
		num_outputs = len(config.output_keys)

		# exit if it is not possible to create any new connections
		if num_connections == utility.max_num_edges(num_nodes) - (utility.max_num_edges(num_inputs) + utility.max_num_edges(num_outputs)):
			return

		while True:
			# pick two random nodes
			node1_key = random.choice(node_keys[num_inputs + num_outputs:])
			node2_key = random.choice(node_keys)

			# try again if chosen nodes are the same and self loops are disabled
			if config.disable_self_loops and node1_key == node2_key:
				continue

			# try again if there is already an existing connection between chosen nodes
			existing_connections = [c for c in self.connections.values() if c.from_key == node1_key and c.to_key == node2_key or c.from_key == node2_key and c.to_key == node1_key]
			if existing_connections:
				continue

			# switch node positions if adding this link would make network recurrent
			if node2_key in config.input_keys or utility.check_if_path_exists_by_connections(node2_key, node1_key, self.connections) or (node2_key, node1_key) in generation_new_connections:
				temp = node1_key
				node1_key = node2_key
				node2_key = temp

			# assign new innovation number or assign existing if structural innovation has already occurred
			key_pair = (node1_key, node2_key)
			if key_pair in generation_new_connections:
				innovation_number = generation_new_connections[key_pair]
			else:
				innovation_number = config.innovation_number
				generation_new_connections[key_pair] = innovation_number
				config.innovation_number += 1

			# create new connection with random weight
			new_connection = Connection(innovation_number, node1_key, node2_key, random.gauss(config.weight_new_mu, config.weight_new_sigma), True)
			self.connections[innovation_number] = new_connection
			return

	def new_node(self, generation_new_nodes, generation_new_connections):
		connections_values = list(self.connections.values())
		# choose a random connection and disable it
		connection = random.choice(connections_values)
		connection.enabled = False

		key_pair = (connection.from_key, connection.to_key)
		# if a node has already been added between these two nodes assign existing innovation numbers and node key
		if key_pair in generation_new_nodes:
			new_node_key = generation_new_nodes[key_pair]

			innovation_number1 = generation_new_connections[(connection.from_key, new_node_key)]

			innovation_number2 = generation_new_connections[(new_node_key, connection.to_key)]
		# otherwise create new and remember structural innovation
		else:
			new_node_key = config.next_node_key
			generation_new_nodes[key_pair] = new_node_key
			config.next_node_key += 1

			innovation_number1 = config.innovation_number
			generation_new_connections[(connection.from_key, new_node_key)] = innovation_number1
			config.innovation_number += 1

			innovation_number2 = config.innovation_number
			generation_new_connections[(new_node_key, connection.to_key)] = innovation_number2
			config.innovation_number += 1

		# create a new node with random bias
		bias = random.gauss(config.bias_new_mu, config.bias_new_sigma)
		new_node = Node(new_node_key, bias)
		self.nodes[new_node_key] = new_node

		# create a new connection and set it's value to 1.0
		new_connection1 = Connection(innovation_number1, connection.from_key, new_node_key, 1.0, True)
		self.connections[innovation_number1] = new_connection1

		# create a new connection and set it's value to the value of disabled connection
		new_connection2 = Connection(innovation_number2, new_node_key, connection.to_key, connection.weight, True)
		self.connections[innovation_number2] = new_connection2

	def duplicate(self):
		clone = copy.deepcopy(self)

		clone.fitness = None
		clone.adjusted_fitness = None

		return clone


def crossover(parent1, parent2):
	# set fitter parent as parent1
	if parent1.fitness > parent2.fitness:
		fitter_parent, other_parent = (parent1, parent2)
	elif parent1.fitness == parent2.fitness:
		# if they have the same fitness, set the one with simpler structure as the fitter parent
		if len(parent1.connections) < len(parent2.connections):
			fitter_parent, other_parent = (parent1, parent2)
		else:
			fitter_parent, other_parent = (parent2, parent1)
	else:
		fitter_parent, other_parent = (parent2, parent1)

	child_connections = {}
	child_nodes = {}

	# iterate through all connections from fitter parent
	for innovation_number, connection in fitter_parent.connections.items():
		# if other parent has the same gene assign a random parent's gene to the child
		if innovation_number in other_parent.connections:
			other_connection = other_parent.connections[innovation_number]

			if random.random() < 0.5:
				new_connection = copy.deepcopy(connection)
			else:
				new_connection = copy.deepcopy(other_connection)

			if not connection.enabled or not other_connection.enabled:
				new_connection.enabled = random.random() > config.stay_disabled_probability

			child_connections[innovation_number] = new_connection
		# otherwise just copy fitter parent's gene
		else:
			new_connection = copy.deepcopy(connection)

			if not connection.enabled:
				new_connection.enabled = random.random() > config.stay_disabled_probability

			child_connections[innovation_number] = new_connection

	# go through all nodes from the fitter parent
	for key, node in fitter_parent.nodes.items():
		# if other parent has the same node choose a random one
		if key in other_parent.nodes:
			other_node = other_parent.nodes[key]

			if random.random() < 0.5:
				new_node = copy.deepcopy(node)
			else:
				new_node = copy.deepcopy(other_node)

			child_nodes[key] = new_node
		# otherwise copy fitter parent's node
		else:
			new_node = copy.deepcopy(node)
			child_nodes[key] = new_node

	# create a new child with chosen connections and nodes
	child = Individual(child_connections, child_nodes)
	return child
