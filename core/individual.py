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
		for key in range(config.num_starting_nodes):
			bias = random.gauss(config.bias_new_mu, config.bias_new_sigma)
			node = Node(key, bias)
			self.nodes[key] = node

		next_innovation_number = 0
		if config.num_starting_hidden_nodes == 0:
			for input_key in config.input_keys:
				for output_key in config.output_keys:
					new_connection = Connection(next_innovation_number, input_key, output_key, random.gauss(config.weight_new_mu, config.weight_new_sigma), True)
					self.connections[next_innovation_number] = new_connection
					next_innovation_number += 1
		else:
			for hidden_node_key in range(config.num_starting_nodes - config.num_starting_hidden_nodes, config.num_starting_nodes):
				for input_key in config.input_keys:
					new_connection = Connection(next_innovation_number, input_key, hidden_node_key, random.gauss(config.weight_new_mu, config.weight_new_sigma), True)
					self.connections[next_innovation_number] = new_connection
					next_innovation_number += 1

				for output_key in config.output_keys:
					new_connection = Connection(next_innovation_number, hidden_node_key, output_key, random.gauss(config.weight_new_mu, config.weight_new_sigma), True)
					self.connections[next_innovation_number] = new_connection
					next_innovation_number += 1

	def mutate(self, generation_new_nodes, generation_new_connections):
		if random.random() < config.new_connection_probability:
			self.new_connection(generation_new_connections)

		if random.random() < config.new_node_probability:
			self.new_node(generation_new_nodes, generation_new_connections)

		self.mutate_connections()
		self.mutate_nodes()

	def mutate_connections(self):
		for connection in self.connections.values():
			r = random.random()
			if r < config.weight_perturbation_probability:
				connection.weight += random.gauss(config.weight_step_mu, config.weight_step_sigma)
			elif r < config.weight_perturbation_probability + config.weight_replace_probability:
				connection.weight = random.gauss(config.weight_new_mu, config.weight_new_sigma)

	def mutate_nodes(self):
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

		if num_connections == utility.max_num_edges(num_nodes) - (
				utility.max_num_edges(num_inputs) + utility.max_num_edges(num_outputs)):
			return

		while True:
			node1_key = random.choice(node_keys[num_inputs + num_outputs:])
			node2_key = random.choice(node_keys)

			if config.disable_self_loops and node1_key == node2_key:
				continue

			existing_connections = [c for c in self.connections.values() if c.from_key == node1_key and c.to_key == node2_key or c.from_key == node2_key and c.to_key == node1_key]
			if existing_connections:
				continue

			if node2_key in config.input_keys or utility.check_if_path_exists_by_connections(node2_key, node1_key, self.connections) or (node2_key, node1_key) in generation_new_connections:
				temp = node1_key
				node1_key = node2_key
				node2_key = temp

			key_pair = (node1_key, node2_key)
			if key_pair in generation_new_connections:
				innovation_number = generation_new_connections[key_pair]
			else:
				innovation_number = config.innovation_number
				generation_new_connections[key_pair] = innovation_number
				config.innovation_number += 1

			new_connection = Connection(innovation_number, node1_key, node2_key, random.gauss(config.weight_new_mu, config.weight_new_sigma), True)
			self.connections[innovation_number] = new_connection
			return

	def new_node(self, generation_new_nodes, generation_new_connections):
		connections_values = list(self.connections.values())
		connection = random.choice(connections_values)
		connection.enabled = False

		key_pair = (connection.from_key, connection.to_key)
		if key_pair in generation_new_nodes:
			new_node_key = generation_new_nodes[key_pair]

			innovation_number1 = generation_new_connections[(connection.from_key, new_node_key)]

			innovation_number2 = generation_new_connections[(new_node_key, connection.to_key)]
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

		bias = random.gauss(config.bias_new_mu, config.bias_new_sigma)
		new_node = Node(new_node_key, bias)
		self.nodes[new_node_key] = new_node

		new_connection1 = Connection(innovation_number1, connection.from_key, new_node_key, 1.0, True)
		self.connections[innovation_number1] = new_connection1

		new_connection2 = Connection(innovation_number2, new_node_key, connection.to_key, connection.weight, True)
		self.connections[innovation_number2] = new_connection2

	def duplicate(self):
		clone = copy.deepcopy(self)

		clone.fitness = None
		clone.adjusted_fitness = None

		return clone


def crossover(parents):
	if parents[0].fitness > parents[1].fitness:
		fitter_parent, other_parent = (parents[0], parents[1])
	elif parents[0].fitness == parents[1].fitness:
		if len(parents[0].connections) < len(parents[1].connections):
			fitter_parent, other_parent = (parents[0], parents[1])
		else:
			fitter_parent, other_parent = (parents[1], parents[0])
	else:
		fitter_parent, other_parent = (parents[1], parents[0])

	child_connections = {}
	child_nodes = {}

	for innovation_number, connection in fitter_parent.connections.items():
		if innovation_number in other_parent.connections:
			other_connection = other_parent.connections[innovation_number]

			if random.random() < 0.5:
				new_connection = copy.deepcopy(connection)
			else:
				new_connection = copy.deepcopy(other_connection)

			if not connection.enabled or not other_connection.enabled:
				new_connection.enabled = random.random() > config.stay_disabled_probability

			child_connections[innovation_number] = new_connection
		else:
			new_connection = copy.deepcopy(connection)

			if not connection.enabled:
				new_connection.enabled = random.random() > config.stay_disabled_probability

			child_connections[innovation_number] = new_connection

	for key, node in fitter_parent.nodes.items():
		if key in other_parent.nodes:
			other_node = other_parent.nodes[key]

			if random.random() < 0.5:
				new_node = copy.deepcopy(node)
			else:
				new_node = copy.deepcopy(other_node)

			child_nodes[key] = new_node
		else:
			new_node = copy.deepcopy(node)
			child_nodes[key] = new_node

	child = Individual(child_connections, child_nodes)

	return child
