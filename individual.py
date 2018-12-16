from config import config
from connection import Connection
import utility
import copy
import random


class Individual:
	def __init__(self, connections=None, nodes=None):
		self.nodes = set()
		self.connections = {}
		self.fitness = None
		self.adjusted_fitness = None

		if connections and nodes:
			self.connections = connections
			self.nodes = nodes
		else:
			self.configure_new()

	def configure_new(self):
		self.nodes.update(range(config.num_starting_nodes))

		next_innovation_number = 0
		if config.num_starting_hidden_nodes == 0:
			for input_key in config.input_keys:
				for output_key in config.output_keys:
					new_connection = Connection(next_innovation_number, input_key, output_key, random.gauss(config.new_mu, config.new_sigma), True)
					self.connections[next_innovation_number] = new_connection
					next_innovation_number += 1
		else:
			for hidden_node_key in range(config.num_starting_nodes - config.num_starting_hidden_nodes, config.num_starting_nodes):
				for input_key in config.input_keys:
					new_connection = Connection(next_innovation_number, input_key, hidden_node_key, random.gauss(config.new_mu, config.new_sigma), True)
					self.connections[next_innovation_number] = new_connection
					next_innovation_number += 1

				for output_key in config.output_keys:
					new_connection = Connection(next_innovation_number, hidden_node_key, output_key, random.gauss(config.new_mu, config.new_sigma), True)
					self.connections[next_innovation_number] = new_connection
					next_innovation_number += 1

	def mutate(self, generation_new_nodes, generation_new_connections):
		if random.random() < config.connection_mutation_probability:
			self.mutate_connections()

		if config.fixed_topology:
			return

		if random.random() < config.new_connection_probability:
			self.new_connection(generation_new_connections)

		if random.random() < config.new_node_probability:
			self.new_node(generation_new_nodes, generation_new_connections)

	def mutate_connections(self):
		for connection in self.connections.values():
			if random.random() < config.perturbation_probability:
				connection.weight = connection.weight + random.gauss(config.step_mu, config.step_sigma)
			else:
				connection.weight = random.random() * 2 - 1

	def new_connection(self, generation_new_connections):
		nodes_list = list(self.nodes)
		num_nodes = len(nodes_list)
		num_connections = len(self.connections.values())
		num_inputs = len(config.input_keys)
		num_outputs = len(config.output_keys)

		if num_connections == utility.max_num_edges(num_nodes) - (utility.max_num_edges(num_inputs) + utility.max_num_edges(num_outputs)):
			return

		while True:
			node1_key = random.choice(nodes_list[num_inputs + num_outputs:])
			node2_key = random.choice(nodes_list)

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

			new_connection = Connection(innovation_number, node1_key, node2_key, random.gauss(config.new_mu, config.new_sigma), True)
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

		self.nodes.add(new_node_key)

		new_connection1 = Connection(innovation_number1, connection.from_key, new_node_key, 1.0, True)
		self.connections[innovation_number1] = new_connection1

		new_connection2 = Connection(innovation_number2, new_node_key, connection.to_key, connection.weight, True)
		self.connections[innovation_number2] = new_connection2


def crossover(parents):
	if parents[0].adjusted_fitness > parents[1].adjusted_fitness:
		fitter_parent, other_parent = (parents[0], parents[1])
	else:
		fitter_parent, other_parent = (parents[1], parents[0])

	innovation_numbers1 = fitter_parent.connections.keys()
	innovation_numbers2 = other_parent.connections.keys()
	all_innovation_numbers = set(innovation_numbers1).union(set(innovation_numbers2))

	child_connections = {}
	child_nodes = set()

	for innovation_number in all_innovation_numbers:
		if innovation_number in fitter_parent.connections and innovation_number in other_parent.connections:
			connection1 = fitter_parent.connections[innovation_number]
			connection2 = other_parent.connections[innovation_number]

			if random.random() < 0.5:
				new_connection = copy.deepcopy(connection1)
			else:
				new_connection = copy.deepcopy(connection2)

			if not connection1.enabled or not connection2.enabled:
				new_connection.enabled = random.random() > config.stay_disabled_probability

			child_connections[innovation_number] = new_connection

			for node_key in [new_connection.from_key, new_connection.to_key]:
				if node_key not in child_nodes:
					child_nodes.add(node_key)

		elif innovation_number in fitter_parent.connections and innovation_number not in other_parent.connections:
			connection = fitter_parent.connections[innovation_number]

			new_connection = copy.deepcopy(connection)

			if not connection.enabled:
				new_connection.enabled = random.random() > config.stay_disabled_probability

			child_connections[innovation_number] = new_connection

			for node_key in [new_connection.from_key, new_connection.to_key]:
				if node_key not in child_nodes:
					child_nodes.add(node_key)

	child = Individual(child_connections, child_nodes)

	return child
