from core.config import config
import math
import numpy as np


def sigmoid(x):
	try:
		return 1 / (1 + math.exp(-config.sigmoid_coef * x))
	except OverflowError:
		return 0


def max_num_edges(num_nodes):
	return num_nodes * (num_nodes - 1) / 2


def check_if_path_exists_by_connections(from_node, to_node, connections, checked=None):
	if not checked:
		checked = {}

	if from_node == to_node:
		return True

	for connection in connections.values():
		if not connection.enabled:
			continue

		if connection.from_key == from_node and connection.to_key == to_node:
			return True

	checked[from_node] = True

	for connection in connections.values():
		if not connection.enabled:
			continue

		if connection.from_key == from_node and connection.to_key not in checked \
			and check_if_path_exists_by_connections(connection.to_key, to_node, connections, checked):
			return True

	return False


def check_if_path_exists_by_neurons(from_key, to_key, neurons, checked=None):
	if not checked:
		checked = {}

	if from_key == to_key or to_key in neurons[from_key].outgoing_keys:
		return True

	checked[from_key] = True

	for key in neurons[from_key].outgoing_keys:
		if key not in checked and check_if_path_exists_by_neurons(key, to_key, neurons, checked):
			return True

	return False


def distance(individual1, individual2):
	connections1 = individual1.connections
	connections2 = individual2.connections

	max_innovation_number1 = max(individual1.connections.keys())
	max_innovation_number2 = max(individual2.connections.keys())
	max_common_innovation_number = min(max_innovation_number1, max_innovation_number2)

	all_innovation_numbers = set(connections1.keys()).union(set(connections2.keys()))

	weight_diffs = []
	E = 0
	D = 0
	max_num_nodes = max(len(connections1), len(connections2))
	max_num_hidden = max_num_nodes - config.num_starting_nodes
	N = max_num_hidden if config.normalize else 1.

	for innovation_number in all_innovation_numbers:
		if innovation_number in connections1 and innovation_number in connections2:
			weight_diff = abs(connections1[innovation_number].weight - connections2[innovation_number].weight)
			weight_diffs.append(weight_diff)
		else:
			if innovation_number > max_common_innovation_number:
				E += 1
			else:
				D += 1

	adjusted_E = (config.c1 * E) / N
	adjusted_D = (config.c2 * D) / N
	avg_weight_diff = sum(weight_diffs) / len(weight_diffs)
	adjusted_weight_diff = config.c3 * avg_weight_diff
	delta = adjusted_E + adjusted_D + adjusted_weight_diff
	return delta, adjusted_D, adjusted_E, adjusted_weight_diff


# from range [0, 1] to [low, high]
def scale(data, low, high):
	return data * np.abs(high - low) + low


# from range [low, high] to [0, 1]
def normalize(data, low, high):
	return (data - low) / np.abs(high - low)


# rescale data to have a mean 0 and stdev 1
def standardize(data, mean, stdev):
	return (data - mean) / stdev
