from core.config import config
import numpy as np


class Statistics:
	def __init__(self):
		# over generations
		self.best_fitnesses = []
		self.avg_fitnesses = []
		self.stdev_fitnesses = []
		self.avg_num_hidden_nodes = []
		self.stdev_num_hidden_nodes = []
		self.avg_num_connections = []
		self.stdev_num_connections = []
		self.species_sizes = []
		self.compatibility_thresholds = []
		self.avg_Es = []
		self.avg_Ds = []
		self.avg_weight_diffs = []

		# over runs
		self.num_evaluations = []
		self.num_hidden_nodes = []
		self.num_connections = []

	def update_fitnesses(self, species):
		fitnesses = []
		for spec in species:
			fitnesses += [individual.fitness for individual in spec.individuals]

		best = max(fitnesses)
		self.best_fitnesses.append(best)

		avg = sum(fitnesses) / len(fitnesses)
		self.avg_fitnesses.append(avg)

		stdev = np.std(fitnesses)
		self.stdev_fitnesses.append(stdev)

	def update_structures(self, species):
		num_hidden_nodes = []
		num_connections = []
		for spec in species:
			num_hidden_nodes += [len(individual.nodes) - config.num_starting_nodes for individual in spec.individuals]
			num_connections += [len(individual.connections) for individual in spec.individuals]

		avg_num_hidden = np.mean(num_hidden_nodes)
		self.avg_num_hidden_nodes.append(avg_num_hidden)

		stdev_num_hidden = np.std(num_hidden_nodes)
		self.stdev_num_hidden_nodes.append(stdev_num_hidden)

		avg_num_conns = np.mean(num_connections)
		self.avg_num_connections.append(avg_num_conns)

		stdev_num_conns = np.std(num_hidden_nodes)
		self.stdev_num_connections.append(stdev_num_conns)

	def update_species(self, species, total_num_species_ever):
		generation_sizes = [0] * total_num_species_ever
		for spec in species:
			generation_sizes[spec.key] = len(spec.individuals)
		self.species_sizes.append(generation_sizes)

		self.compatibility_thresholds.append(config.compatibility_threshold)

	def update_distances(self, Es, Ds, weight_diffs):
		avg_E = sum(Es) / len(Es)
		self.avg_Es.append(avg_E)

		avg_D = sum(Ds) / len(Ds)
		self.avg_Ds.append(avg_D)

		avg_weight_diff = sum(weight_diffs) / len(weight_diffs)
		self.avg_weight_diffs.append(avg_weight_diff)

	def update_run(self, num_ev, best_individual):
		self.num_evaluations.append(num_ev)

		num_hidden = len(best_individual.nodes) - config.num_starting_nodes
		self.num_hidden_nodes.append(num_hidden)

		num_conn = len([conn for conn in best_individual.connections.values() if conn.enabled])
		self.num_connections.append(num_conn)

	def reset_generation(self):
		self.best_fitnesses = []
		self.avg_fitnesses = []
		self.stdev_fitnesses = []
		self.avg_num_hidden_nodes = []
		self.stdev_num_hidden_nodes = []
		self.avg_num_connections = []
		self.stdev_num_connections = []
		self.species_sizes = []
		self.compatibility_thresholds = []
		self.avg_Es = []
		self.avg_Ds = []
		self.avg_weight_diffs = []
