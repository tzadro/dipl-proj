from Config import config
from Node import Node
from Connection import Connection
from Neuron import Neuron
from Phenotype import Phenotype
from Individual import Individual
import helperfunctions
import numpy as np


def test_sigmoid():
	res = helperfunctions.sigmoid(2.5)
	print('Sigmoid test:', round(res, 2) == 0.92)


def test_check_if_path_exists(connections):
	from_node = 5
	to_node = 8
	res1 = helperfunctions.check_if_path_exists(from_node, to_node, connections)

	from_node = 8
	to_node = 5
	res2 = helperfunctions.check_if_path_exists(from_node, to_node, connections)

	print('Check_if_path_exists test:', res1 is True and res2 is False)


def test_check_if_path_exists2(connections):
	phenotype = Phenotype(connections)

	from_node = 5
	to_node = 8
	res1 = helperfunctions.check_if_path_exists2(from_node, to_node, phenotype.neurons)

	from_node = 8
	to_node = 5
	res2 = helperfunctions.check_if_path_exists2(from_node, to_node, phenotype.neurons)

	print('Check_if_path_exists2 test:', res1 is True and res2 is False)


def test_distance(individual1, individual2):
	config.c1 = 8
	config.c2 = 4
	config.c3 = 1

	res = helperfunctions.distance(individual1, individual2)

	print('Distance test:', round(res, 2) == 2.3)


def test_innovation_numbers_union(connections1, connections2):
	union = helperfunctions.innovation_numbers_union(connections1, connections2)

	res = True
	for innovation_number in range(18):
		if innovation_number not in union:
			res = False
			break

	print('Innovation_numbers_union test:', res)


def test_phenotype(connections):
	config.input_keys = [0, 1, 2]
	config.output_keys = [3, 4]
	config.action_space_discrete = False
	config.action_space_high = np.array([2., 1.])
	config.action_space_low = np.array([-1., -1.])

	inputs = [1., 1., 1.]
	phenotype = Phenotype(connections)
	res = phenotype.forward(inputs)

	print('Phenotype test:', round(res[0], 2) == 1.07 and round(res[1], 2) == 0.39)


def test_neuron():
	connections = {
		0: Connection(0, 0, 2, 0.2, True),
		1: Connection(1, 1, 2, 0.3, True),
		2: Connection(2, 2, 3, 0.4, True),
		3: Connection(3, 2, 2, 0.5, True)
	}
	neurons = {
		0: Neuron(0),
		1: Neuron(1),
		2: Neuron(2),
		3: Neuron(3)
	}

	neurons[0].add_outgoing(2)
	neurons[1].add_outgoing(2)
	return_path_exists = helperfunctions.check_if_path_exists2(connections[0].to_key, connections[0].from_key, neurons)
	neurons[2].add_incoming(connections[0], return_path_exists)
	return_path_exists = helperfunctions.check_if_path_exists2(connections[1].to_key, connections[1].from_key, neurons)
	neurons[2].add_incoming(connections[1], return_path_exists)
	return_path_exists = helperfunctions.check_if_path_exists2(connections[3].to_key, connections[3].from_key, neurons)
	neurons[2].add_incoming(connections[3], return_path_exists)
	neurons[2].add_outgoing(3)
	return_path_exists = helperfunctions.check_if_path_exists2(connections[2].to_key, connections[2].from_key, neurons)
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

	print('Neuron test:', round(res1, 2) == 0.56 and round(res2, 2) == 0.57)


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
		0: Node(0),
		1: Node(1),
		2: Node(2),
		3: Node(3),
		4: Node(4),
		5: Node(5),
		6: Node(6),
		7: Node(7),
		8: Node(8)
	}
	next_new_innovation1 = 16
	next_new_node1 = 9
	individual1 = Individual(connections1, nodes1, next_new_innovation1, next_new_node1)

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
		0: Node(0),
		1: Node(1),
		2: Node(2),
		3: Node(3),
		4: Node(4),
		5: Node(5),
		6: Node(6),
		8: Node(8)
	}
	next_new_innovation2 = 18
	next_new_node2 = 9
	individual2 = Individual(connections2, nodes2, next_new_innovation2, next_new_node2)

	test_sigmoid()
	test_check_if_path_exists(connections1)
	test_check_if_path_exists2(connections1)
	test_distance(individual1, individual2)
	test_innovation_numbers_union(connections1, connections2)
	test_neuron()
	test_phenotype(connections1)


run()
