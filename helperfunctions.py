from Config import config
import math


def sigmoid(x):
	return 1 / (1 + math.exp(-x))


def max_num_edges(num_nodes):
	return num_nodes * (num_nodes - 1) / 2


def check_if_path_exists(from_node, to_node, connections, checked):
	for connection in connections.values():
		if not connection.enabled:
			continue

		if connection.from_key == from_node and connection.to_key == to_node:
			return True

	checked[from_node] = True

	for connection in connections.values():
		if not connection.enabled:
			continue

		if connection.from_key == from_node and connection.to_key not in checked and check_if_path_exists(
				connection.to_key, to_node, connections, checked):
			return True

	return False


def check_if_path_exists2(from_key, to_key, neurons, checked):
	if to_key in neurons[from_key].outgoing_keys:
		return True

	checked[from_key] = True

	for key in neurons[from_key].outgoing_keys:
		if key not in checked and check_if_path_exists2(key, to_key, neurons, checked):
			return True

	return False


# todo: are disabled genes included?
def distance(individual1, individual2):
	connections1 = individual1.connections
	connections2 = individual2.connections

	innovation_numbers = innovation_numbers_union(connections1, connections2)
	max_common = min(individual1.max_innovation, individual1.max_innovation)

	weight_diffs = []
	E = 0
	D = 0
	N = max(len(connections1), len(connections2))

	for innovation_number in innovation_numbers:
		if innovation_number in connections1 and innovation_number in connections2:
			weight_diff = abs(connections1[innovation_number].weight - connections2[innovation_number].weight)
			weight_diffs.append(weight_diff)
		else:
			if innovation_number > max_common:
				E = E + 1
			else:
				D = D + 1

	delta = (config.c1 * E) / N + (config.c2 * D) / N + config.c3 * sum(weight_diffs) / len(weight_diffs)
	return delta


def innovation_numbers_union(connections1, connections2):
	innovation_numbers1 = [connection.innovation_number for connection in connections1.values()]
	innovation_numbers2 = [connection.innovation_number for connection in connections2.values()]

	innovation_numbers = set().union(innovation_numbers1).union(innovation_numbers2)
	return innovation_numbers

