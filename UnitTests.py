from config import config
from connection import Connection
from node import Node
from neuron import Neuron
from phenotype import Phenotype
from individual import Individual
from interface import NetworkVisualizer
import utility
import numpy as np


def test_sigmoid():
	config.sigmoid_coef = 1

	res = utility.sigmoid(2.5)
	print('Sigmoid test:', round(res, 2) == 0.92)


def test_check_if_path_exists_by_connections(connections):
	from_node = 5
	to_node = 8
	res1 = utility.check_if_path_exists_by_connections(from_node, to_node, connections)

	from_node = 8
	to_node = 5
	res2 = utility.check_if_path_exists_by_connections(from_node, to_node, connections)

	print('Check_if_path_exists_by_connections test:', res1 is True and res2 is False)


def test_check_if_path_exists_by_neurons(connections, nodes):
	config.input_keys = [0, 1, 2]
	config.output_keys = [3, 4]

	phenotype = Phenotype(connections.values(), nodes.values(), config)

	from_node = 5
	to_node = 8
	res1 = utility.check_if_path_exists_by_neurons(from_node, to_node, phenotype.neurons)

	from_node = 8
	to_node = 5
	res2 = utility.check_if_path_exists_by_neurons(from_node, to_node, phenotype.neurons)

	print('Check_if_path_exists_by_neurons test:', res1 is True and res2 is False)


def test_distance(individual1, individual2):
	config.c1 = 8
	config.c2 = 4
	config.c3 = 1
	config.normalize = True

	res = utility.distance(individual1, individual2)

	print('Distance test:', round(res, 2) == 2.3)


def test_neuron():
	config.sigmoid_coef = 1

	connections = {
		0: Connection(0, 0, 2, 0.2, True),
		1: Connection(1, 1, 2, 0.3, True),
		2: Connection(2, 2, 3, 0.4, True),
		3: Connection(3, 2, 2, 0.5, True)
	}
	neurons = {
		0: Neuron(0, 0),
		1: Neuron(1, 0),
		2: Neuron(2, 0.2),
		3: Neuron(3, 0.3)
	}

	neurons[0].add_outgoing(2)
	neurons[1].add_outgoing(2)
	return_path_exists = utility.check_if_path_exists_by_neurons(connections[0].to_key, connections[0].from_key, neurons)
	neurons[2].add_incoming(connections[0], return_path_exists)
	return_path_exists = utility.check_if_path_exists_by_neurons(connections[1].to_key, connections[1].from_key, neurons)
	neurons[2].add_incoming(connections[1], return_path_exists)
	return_path_exists = utility.check_if_path_exists_by_neurons(connections[3].to_key, connections[3].from_key, neurons)
	neurons[2].add_incoming(connections[3], return_path_exists)
	neurons[2].add_outgoing(3)
	return_path_exists = utility.check_if_path_exists_by_neurons(connections[2].to_key, connections[2].from_key, neurons)
	neurons[3].add_incoming(connections[2], return_path_exists)

	for neuron in neurons.values():
		neuron.reset()

	neurons[0].set_value(1., neurons)
	neurons[1].set_value(1., neurons)
	res1 = neurons[3].value

	for neuron in neurons.values():
		neuron.reset()

	neurons[0].set_value(1., neurons)
	neurons[1].set_value(1., neurons)
	res2 = neurons[3].value

	print('Neuron test:', round(res1, 4) == 0.6381 and round(res2, 4) == 0.6445)


def test_phenotype(connections, nodes):
	config.sigmoid_coef = 1
	config.input_keys = [0, 1, 2]
	config.output_keys = [3, 4]

	action_space_high = np.array([2., 1.])
	action_space_low = np.array([-1., -1.])

	inputs = [1., 1., 1.]
	phenotype = Phenotype(connections.values(), nodes.values(), config)
	output = phenotype.forward(inputs)
	res = utility.scale(output, action_space_low, action_space_high)

	print('Phenotype test:', round(res[0], 2) == 1.07 and round(res[1], 2) == 0.39)


def test_interface():
	config.input_keys = [0, 1]
	config.output_keys = [2, 3]

	connections = {
		0: Connection(0, 0, 4, 0.2, True),
		1: Connection(1, 1, 4, 0.3, True),
		2: Connection(2, 4, 2, 0.4, True),
		3: Connection(3, 4, 3, 0.5, True),
		4: Connection(4, 4, 4, 0.3, True)
	}
	nodes = {
		0: Node(0, 0),
		1: Node(1, 0),
		2: Node(2, 0),
		3: Node(3, 0),
		4: Node(4, 0)
	}

	network_visualizer = NetworkVisualizer()
	network_visualizer.update_node_positions(connections, nodes)
	network_visualizer.visualize_network(connections)


def run():
	connections1 = {
		0: Connection(0, 0, 6, 0.2, True),
		1: Connection(1, 0, 5, 0.3, True),
		2: Connection(2, 1, 5, 0.4, True),
		3: Connection(3, 1, 8, 0.5, True),
		4: Connection(4, 2, 7, 0.3, True),
		5: Connection(5, 2, 8, 0.7, True),
		6: Connection(6, 5, 6, 0.2, True),
		7: Connection(7, 5, 7, 0.1, True),
		8: Connection(8, 7, 6, 0.1, True),
		9: Connection(9, 6, 8, 0.3, True),
		10: Connection(10, 6, 3, 0.6, True),
		11: Connection(11, 6, 4, 0.7, True),
		12: Connection(12, 5, 3, 0.3, True),
		13: Connection(13, 7, 4, 0.7, True),
		14: Connection(14, 8, 3, 0.3, True),
		15: Connection(15, 8, 5, 0.8, False)
	}
	nodes1 = {
		0: Node(0, 0),
		1: Node(1, 0),
		2: Node(2, 0),
		3: Node(3, 0),
		4: Node(4, 0),
		5: Node(5, 0),
		6: Node(6, 0),
		7: Node(7, 0),
		8: Node(8, 0)
	}
	individual1 = Individual(connections1, nodes1)

	connections2 = {
		0: Connection(0, 0, 6, 0.9, True),
		1: Connection(1, 0, 5, 0.4, True),
		2: Connection(2, 1, 5, 0.7, True),
		3: Connection(3, 1, 8, 0.6, True),
		5: Connection(5, 2, 8, 0.4, True),
		6: Connection(6, 5, 6, 0.7, True),
		9: Connection(9, 6, 8, 0.1, True),
		10: Connection(10, 6, 3, 0.4, True),
		11: Connection(11, 6, 4, 0.1, True),
		12: Connection(12, 5, 3, 0.4, True),
		14: Connection(14, 8, 3, 0.5, True),
		15: Connection(15, 8, 5, 0.5, False),
		16: Connection(16, 1, 4, 0.5, True),
		17: Connection(17, 2, 4, 0.8, True)
	}
	nodes2 = {
		0: Node(0, 0),
		1: Node(1, 0),
		2: Node(2, 0),
		3: Node(3, 0),
		4: Node(4, 0),
		5: Node(5, 0),
		6: Node(6, 0),
		7: Node(7, 0),
		8: Node(8, 0)
	}
	individual2 = Individual(connections2, nodes2)

	test_sigmoid()
	test_check_if_path_exists_by_connections(connections1)
	test_check_if_path_exists_by_neurons(connections1, nodes1)
	test_distance(individual1, individual2)
	test_neuron()
	test_phenotype(connections1, nodes1)
	test_interface()


run()
