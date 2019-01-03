from core.config import config
import numpy as np


class Statistics:
	def __init__(self):
		# over generations
		self.best_fitnesses = []
		self.avg_fitnesses = []
		self.stdev_fitnesses = []
		self.species_sizes = []

		# over runs
		self.num_evaluations = []
		self.num_hidden_nodes = []
		self.num_connections = []

	def update_generation(self, population):
		fitnesses = [individual.fitness for individual in population.individuals]

		best = max(fitnesses)
		self.best_fitnesses.append(best)

		avg = sum(fitnesses) / len(fitnesses)
		self.avg_fitnesses.append(avg)

		stdev = np.std(fitnesses)
		self.stdev_fitnesses.append(stdev)

		num_species = population.next_species_key
		generation_sizes = [0] * num_species
		for spec in population.species:
			generation_sizes[spec.key] = len(spec.individuals)
		self.species_sizes.append(generation_sizes)

	def update_run(self, num_ev, best_individual):
		self.num_evaluations.append(num_ev)

		num_hidden = len(best_individual.nodes) - (config.num_starting_nodes + config.num_starting_hidden_nodes)
		self.num_hidden_nodes.append(num_hidden)

		num_conn = len([conn for conn in best_individual.connections.values() if conn.enabled])
		self.num_connections.append(num_conn)

	def reset_generation(self):
		self.best_fitnesses = []
		self.avg_fitnesses = []
		self.stdev_fitnesses = []
		self.species_sizes = []
