from Config import config
from Connection import Connection
from Node import Node
from Phenotype import Phenotype
import helperfunctions
import copy
import random


class Individual:  # Genome
	def __init__(self, connections=None, nodes=None, next_new_innovation=None, next_new_node=None):
		self.nodes = {}
		self.connections = {}
		self.fitness = None
		self.adjusted_fitness = None
		self.next_new_innovation = None
		self.next_new_node = None

		if connections is None or nodes is None or next_new_innovation is None or next_new_node is None:
			self.configure_new()
		else:
			self.connections = connections
			self.nodes = nodes
			self.next_new_innovation = next_new_innovation
			self.next_new_node = next_new_node

	def configure_new(self):
		self.next_new_node = 0
		self.next_new_innovation = 0

		num_starting_nodes = len(config.input_keys) + len(config.output_keys)
		for _ in range(num_starting_nodes):
			self.nodes[self.next_new_node] = Node(self.next_new_node)
			self.next_new_node += 1

		for input_key in config.input_keys:
			for output_key in config.output_keys:
				new_connection = Connection(self.next_new_innovation, input_key, output_key, random.gauss(config.new_mu, config.new_sigma), True)
				self.connections[self.next_new_innovation] = new_connection
				self.next_new_innovation = self.next_new_innovation + 1

	def evaluate_fitness(self, env):
		phenotype = Phenotype(self.connections)

		observation = env.reset()

		self.fitness = 6  # todo: should be 0, set this way so the score is never less than 0 in Pixelcopter game (minimum is -5)
		while True:
			output = phenotype.forward(observation)
			observation, reward, done, info = env.step(output)

			self.fitness += reward

			if done:
				return self.fitness

	def mutate(self, generation_innovations):
		if random.random() < config.connection_mutation_probability:
			self.mutate_connections()

		if config.fixed_topology:
			return

		if random.random() < config.new_connection_probability:
			self.new_connection(generation_innovations)

		if random.random() < config.new_node_probability:
			self.new_node()

	def mutate_connections(self):
		for connection in self.connections.values():
			if random.random() < config.perturbation_probability:
				connection.weight = connection.weight + random.gauss(config.step_mu, config.step_sigma)  # todo: random() * 2 * config.step - config.step ?
			else:
				connection.weight = random.random() * 2 - 1

	def new_connection(self, generation_innovations):
		nodes_list = list(self.nodes.values())
		num_nodes = len(nodes_list)
		num_connections = len(self.connections.values())
		num_inputs = len(config.input_keys)
		num_outputs = len(config.output_keys)

		if num_connections == helperfunctions.max_num_edges(num_nodes) - (helperfunctions.max_num_edges(num_inputs) + helperfunctions.max_num_edges(num_outputs)):
			return

		while True:
			node1 = nodes_list[random.randrange(num_inputs + num_outputs, num_nodes)]
			node2 = nodes_list[random.randrange(num_nodes)]

			# todo: enable this
			if node1.key == node2.key:
				continue

			existing_connections = [c for c in self.connections.values() if c.from_key == node1.key and c.to_key == node2.key or c.from_key == node2.key and c.to_key == node1.key]
			if existing_connections:
				continue

			if node2.key in config.input_keys or helperfunctions.check_if_path_exists(node2.key, node1.key, self.connections) or (node2.key, node1.key) in generation_innovations:
				temp = node1
				node1 = node2
				node2 = temp

			if (node1.key, node2.key) in generation_innovations:
				innovation_number = generation_innovations[(node1.key, node2.key)]
				self.next_new_innovation = max(self.next_new_innovation, innovation_number)
			else:
				innovation_number = config.innovation_number
				generation_innovations[(node1.key, node2.key)] = innovation_number
				self.next_new_innovation = config.innovation_number
				config.innovation_number += 1

			new_connection = Connection(innovation_number, node1.key, node2.key, random.random() * 2 - 1, True)
			self.connections[innovation_number] = new_connection
			return

	def new_node(self):
		connections_values = list(self.connections.values())

		connection = connections_values[random.randrange(len(connections_values))]
		connection.enabled = False

		new_node = Node(config.next_node_key)
		self.next_new_node = config.next_node_key
		config.next_node_key += 1
		self.nodes[new_node.key] = new_node

		new_connection1 = Connection(config.innovation_number, connection.from_key, new_node.key, 1.0, True)
		self.next_new_innovation = config.innovation_number
		self.connections[config.innovation_number] = new_connection1
		config.innovation_number += 1

		new_connection2 = Connection(config.innovation_number, new_node.key, connection.to_key, connection.weight, True)
		self.next_new_innovation = config.innovation_number
		self.connections[config.innovation_number] = new_connection2
		config.innovation_number += 1


# todo: move somewhere?
def crossover(parents):
	parent1 = parents[0]
	parent2 = parents[1]

	parent1_connections = parent1.connections
	parent2_connections = parent2.connections

	innovation_numbers = helperfunctions.innovation_numbers_union(parent1_connections, parent2_connections)

	child_connections = {}
	child_nodes = {}

	for innovation_number in innovation_numbers:
		if innovation_number in parent1_connections and innovation_number in parent2_connections:
			connection1 = parent1_connections[innovation_number]
			connection2 = parent2_connections[innovation_number]

			if random.random() < 0.5:
				new_connection = copy.deepcopy(connection1)
			else:
				new_connection = copy.deepcopy(connection2)

			if not connection1.enabled or not connection2.enabled:
				new_connection.enabled = random.random() < (1 - config.disable_probability)

			child_connections[innovation_number] = new_connection

			if new_connection.from_key not in child_nodes:
				child_nodes[new_connection.from_key] = Node(new_connection.from_key)

			if new_connection.to_key not in child_nodes:
				child_nodes[new_connection.to_key] = Node(new_connection.to_key)

		elif innovation_number in parent1_connections and innovation_number not in parent2_connections:
			connection1 = parent1_connections[innovation_number]
			new_connection = copy.deepcopy(connection1)

			if not connection1.enabled:
				new_connection.enabled = random.random() < (1 - config.disable_probability)

			child_connections[innovation_number] = new_connection

			if new_connection.from_key not in child_nodes:
				child_nodes[new_connection.from_key] = Node(new_connection.from_key)

			if new_connection.to_key not in child_nodes:
				child_nodes[new_connection.to_key] = Node(new_connection.to_key)

		elif innovation_number not in parent1_connections and innovation_number in parent2_connections:
			connection2 = parent2_connections[innovation_number]
			new_connection = copy.deepcopy(connection2)

			if not connection2.enabled:
				new_connection.enabled = random.random() < (1 - config.disable_probability)

			child_connections[innovation_number] = new_connection

			if new_connection.from_key not in child_nodes:
				child_nodes[new_connection.from_key] = Node(new_connection.from_key)

			if new_connection.to_key not in child_nodes:
				child_nodes[new_connection.to_key] = Node(new_connection.to_key)

	next_new_innovation = max(parent1.next_new_innovation, parent2.next_new_innovation)
	next_new_node = max(parent1.next_new_node, parent2.next_new_node)
	child = Individual(child_connections, child_nodes, next_new_innovation, next_new_node)
	return child
