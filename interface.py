from config import config
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class NetworkVisualizer:
	def __init__(self):
		self.node_positions = {}

		num_inputs = len(config.input_keys)
		for i, key in enumerate(config.input_keys):
			x = 0
			if num_inputs == 1:
				y = config.network_canvas_height / 2.
			else:
				y = (num_inputs - (i + 1)) * config.network_canvas_height / (num_inputs - 1)

			self.node_positions[key] = (x, y)

		num_outputs = len(config.output_keys)
		for i, key in enumerate(config.output_keys):
			x = config.network_canvas_width
			if num_outputs == 1:
				y = config.network_canvas_height / 2.
			else:
				y = (num_outputs - (i + 1)) * config.network_canvas_height / (num_outputs - 1)

			self.node_positions[key] = (x, y)

		hidden_start_key = num_inputs + num_outputs
		hidden_keys = range(hidden_start_key, hidden_start_key + config.num_starting_hidden_nodes)
		for i, key in enumerate(hidden_keys):
			x = config.network_canvas_width / 2
			if config.num_starting_hidden_nodes == 1:
				y = config.num_starting_hidden_nodes / 2.
			else:
				y = (config.num_starting_hidden_nodes - (i + 1)) * config.network_canvas_height / (config.num_starting_hidden_nodes - 1)

			self.node_positions[key] = (x, y)

	def visualize_network(self, connections):
		edges = [(connection.from_key, connection.to_key, round(connection.weight, 2)) for connection in connections.values() if connection.enabled]

		G = nx.DiGraph()
		G.add_weighted_edges_from(edges)

		self_loop_keys = [connection.from_key for connection in connections.values() if connection.enabled and connection.from_key == connection.to_key]
		node_colors = ['g' if node_key in self_loop_keys else 'r' for node_key in G]

		nx.draw(G, self.node_positions, node_color=node_colors, with_labels=True)
		labels = nx.get_edge_attributes(G, 'weight')
		nx.draw_networkx_edge_labels(G, self.node_positions, edge_labels=labels)

		plt.show()

	def update_node_positions(self, connections, nodes):
		for key in nodes.keys():
			if key not in self.node_positions:
				neighbor_nodes = [connection.to_key if connection.from_key == key else connection.from_key for connection in connections.values() if (connection.to_key == key or connection.from_key == key) and connection.from_key != connection.to_key]
				x = sum([self.node_positions[node_key][0] for node_key in neighbor_nodes]) / len(neighbor_nodes)
				y = sum([self.node_positions[node_key][1] for node_key in neighbor_nodes]) / len(neighbor_nodes)

				self.node_positions[key] = (x, y)


def log(message):
	if config.verbose:
		print(message)


def verbose(i, population, best_fitness, avg_fitness):
	print('Generation: {:d}, num_individuals: {:d}, best_score: {:.2f}, avg_score: {:.2f}'.format(i, len(population.individuals), best_fitness, avg_fitness))
	print('Num organisms with more than default number of connections: {:d}'.format(sum([len(individual.connections.values()) > config.num_starting_connections for individual in population.individuals])))
	for j, spec in enumerate(population.species):
		print('\tSpecies: {:d}'.format(j))
		print('\t\tfitness: {:.2f}'.format(spec.adjusted_fitness))
		print('\t\tnum_individuals: {:d}, num_children: {:d}'.format(len(spec.individuals), spec.num_children))
		best_adjusted_fitness = spec.individuals[0].adjusted_fitness
		avg_adjusted_fitness = sum([individual.adjusted_fitness for individual in spec.individuals]) / len(spec.individuals)
		print('\t\tbest_adjusted_fitness: {:.2f}, avg_adjusted_fitness: {:.2f}'.format(best_adjusted_fitness, avg_adjusted_fitness))


def plot_overall_fitness(best_fitnesses, avg_fitnesses, stdev_fitnesses):
	generations = range(len(best_fitnesses))
	best = np.array(best_fitnesses)
	avg = np.array(avg_fitnesses)
	stdev = np.array(stdev_fitnesses)

	plt.plot(generations, best, color='red', label='Best score')
	plt.plot(generations, avg, color='blue', label='Average score')
	plt.plot(generations, avg + stdev, color='blue', linestyle='dashed')
	plt.plot(generations, avg - stdev, color='blue', linestyle='dashed')
	plt.title('Fitness over generations')
	plt.xlabel('Generation')
	plt.ylabel('Score')
	plt.legend()
	plt.show()


def plot_species_sizes(species_sizes):
	num_generations = len(species_sizes)
	num_species = len(species_sizes[-1])

	curves = np.zeros((num_species, num_generations))
	for i, row in enumerate(species_sizes):
		for j, element in enumerate(row):
			curves[j][i] = element

	plt.stackplot(range(num_generations), curves)
	plt.title('Species sizes over generations')
	plt.xlabel('Generation')
	plt.ylabel('Size')
	plt.show()


def print_evaluation_stats(num_evaluations, num_hidden_nodes, num_connections):
	num_finished_runs = len(num_evaluations)
	avg_ev = sum(num_evaluations) / num_finished_runs
	avg_gen = avg_ev / config.pop_size
	std_ev = np.std(num_evaluations)
	std_gen = std_ev / config.pop_size
	avg_hidden = sum(num_hidden_nodes) / num_finished_runs
	avg_conn = sum(num_connections) / num_finished_runs

	print('Num evaluations:')
	print('\tavg: {:.2f} (~{:d} generations)'.format(avg_ev, int(avg_gen)))
	print('\tstdev: {:.2f} (~{:d} generations)'.format(std_ev, int(std_gen)))
	print('Structure:')
	print('\tavg num hidden nodes: {:.2f}'.format(avg_hidden))
	print('\tavg num connections: {:.2f}'.format(avg_conn))
	print('From: {:d} finished runs (of {:d} actual runs)'.format(num_finished_runs, config.num_runs))
