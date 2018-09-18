from Config import config
from Connection import Connection
from Node import Node
from Phenotype import Phenotype
import helperfunctions
import copy
import random
import networkx as nx
import matplotlib.pyplot as plt


class Individual:  # Genome
	def __init__(self, connections=None, nodes=None, node_pairs=None, max_innovation=None, max_node=None):
		self.nodes = {}
		self.connections = {}
		self.fitness = 0
		self.max_innovation = 0
		self.max_node = 0
		self.node_pairs = []

		if connections is None or nodes is None or node_pairs is None or max_innovation is None or max_node is None:
			self.configure_new()
		else:
			self.connections = connections
			self.nodes = nodes
			self.node_pairs = node_pairs
			self.max_innovation = max_innovation
			self.max_node = max_node

	def configure_new(self):
		for key in config.input_keys:
			self.nodes[key] = Node(key)
			self.max_node = self.max_node + 1

		for key in config.output_keys:
			self.nodes[key] = Node(key)
			self.max_node = self.max_node + 1

		for input_key in config.input_keys:
			for output_key in config.output_keys:
				new_connection = Connection(self.max_innovation, input_key, output_key, random.gauss(config.new_mu, config.new_sigma), True)
				self.connections[self.max_innovation] = new_connection
				self.max_innovation = self.max_innovation + 1
				self.node_pairs.append((input_key, output_key))

	def evaluate_fitness(self, env):
		phenotype = Phenotype(self.connections)

		if config.visualize:
			self.visualize()

		observation = env.reset()

		while True:
			env.render()

			output_key = phenotype.forward(observation)
			observation, reward, done, info = env.step(config.actions[output_key])
			self.fitness = self.fitness + reward

			if done:
				return self.fitness

	def visualize(self):
		edges = [(connection.from_key, connection.to_key, round(connection.weight, 2)) for connection in self.connections.values()]

		G = nx.DiGraph()
		G.add_weighted_edges_from(edges)
		pos = nx.spring_layout(G)

		nx.draw(G, pos)
		labels = nx.get_edge_attributes(G, 'weight')
		nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
		plt.show()

	def mutate(self, generation_innovations):
		if config.fixed_topology:
			self.mutate_connections()
			return

		if random.random() < config.connection_mutation_probability:
			self.mutate_connections()

		if random.random() < config.new_connection_probability:
			self.new_connection(generation_innovations)

		if random.random() < config.new_node_probability:
			self.new_node()

	def mutate_connections(self):
		for connection in self.connections.values():
			if random.random() < config.perturbation_probability:
				connection.weight = connection.weight + random.gauss(config.step_mu, config.step_sigma)  # random() * 2 * config.step - config.step
			else:
				connection.weight = random.random() * 2 - 1

	def new_connection(self, generation_innovations):
		num_connections = len(self.node_pairs)
		num_nodes = len(list(self.nodes.keys()))
		num_inputs = len(config.input_keys)
		num_outputs = len(config.output_keys)

		node_list = list(self.nodes.values())

		if num_connections == helperfunctions.max_num_edges(num_nodes) - (
				helperfunctions.max_num_edges(num_inputs) + helperfunctions.max_num_edges(num_outputs)):
			return

		while True:
			node1 = node_list[random.randrange(num_inputs + num_outputs, num_nodes)]
			node2 = node_list[random.randrange(num_nodes)]

			if node1.key == node2.key:
				continue

			pair = (min(node1.key, node2.key), max(node1.key, node2.key))

			if pair in self.node_pairs:
				continue

			self.node_pairs.append(pair)

			if node2.key in config.input_keys or helperfunctions.check_if_path_exists(node2.key, node1.key, self.connections, {}) or (node2.key, node1.key) in generation_innovations:
				temp = node1
				node1 = node2
				node2 = temp

			if (node1.key, node2.key) in generation_innovations:
				innovation_number = generation_innovations[(node1.key, node2.key)]
				self.max_innovation = max(self.max_innovation, innovation_number)
			else:
				innovation_number = config.innovation_number
				generation_innovations[(node1.key, node2.key)] = innovation_number
				self.max_innovation = config.innovation_number
				config.innovation_number = config.innovation_number + 1

			new_connection = Connection(innovation_number, node1.key, node2.key, random.random() * 2 - 1, True)
			self.connections[innovation_number] = new_connection
			return

	def new_node(self):
		connections_values = list(self.connections.values())

		connection = connections_values[random.randrange(len(connections_values))]
		connection.enabled = False

		new_node = Node(config.node_key)
		self.max_node = config.node_key
		config.node_key = config.node_key + 1
		self.nodes[new_node.key] = new_node

		new_connection1 = Connection(config.innovation_number, connection.from_key, new_node.key, 1.0, True)
		self.max_innovation = config.innovation_number
		self.connections[config.innovation_number] = new_connection1
		config.innovation_number = config.innovation_number + 1

		new_connection2 = Connection(config.innovation_number, new_node.key, connection.to_key, connection.weight, True)
		self.max_innovation = config.innovation_number
		self.connections[config.innovation_number] = new_connection2
		config.innovation_number = config.innovation_number + 1


# todo: move somewhere?
def crossover(parents):
	parent1 = parents[0]
	parent2 = parents[1]

	parent1_connections = parent1.connections
	parent2_connections = parent2.connections

	innovation_numbers = helperfunctions.innovation_numbers_union(parent1_connections, parent2_connections)

	child_connections = {}
	child_nodes = {}
	node_pairs = []

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

			node_pairs.append((new_connection.from_key, new_connection.to_key))
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

			node_pairs.append((new_connection.from_key, new_connection.to_key))
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

			node_pairs.append((new_connection.from_key, new_connection.to_key))

	max_innovation = max(parent1.max_innovation, parent2.max_innovation)
	max_node = max(parent1.max_node, parent2.max_node)
	child = Individual(child_connections, child_nodes, node_pairs, max_innovation, max_node)
	return child
