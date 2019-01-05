from core.config import config
from core.population import Population
from core.individual import crossover
import math
import random


class AbstractNEAT:
	def __init__(self, evaluate, stats):
		self.evaluate = evaluate
		self.stats = stats
		self.population = Population()

	def epoch(self):
		pass

	def reset(self):
		self.stats.reset_generation()
		self.population = Population()


class NEAT(AbstractNEAT):
	def epoch(self):
		best_individual = self.population.evaluate_fitness(self.evaluate)
		self.stats.update_generation(self.population)

		self.population.adjust_fitness()
		self.population.assign_num_children()

		self.population.remove_worst()
		self.population.breed_new_generation()

		self.population.speciate()

		return best_individual


class NewNEAT(AbstractNEAT):
	def epoch(self):
		# evaluate population
		best_individual = self.population.evaluate_fitness(self.evaluate)
		self.stats.update_generation(self.population)

		# sort by max unadjusted fitness
		for spec in self.population.species:
			spec.individuals.sort(key=lambda x: -x.fitness)
		self.population.species.sort(key=lambda x: -x.individuals[0].fitness)

		# remove stagnant species
		for spec in reversed(self.population.species):
			if len(self.population.species) <= config.species_elitism:
				break

			best_fitness = spec.individuals[0].fitness
			if best_fitness > spec.max_fitness_ever:
				spec.num_generations_before_last_improvement = 0
				spec.max_fitness_ever = best_fitness
			else:
				spec.num_generations_before_last_improvement += 1

				if spec.num_generations_before_last_improvement > config.max_num_generations_before_species_improvement:
					self.population.species.remove(spec)

		# find min and max for normalization
		min_fitness = math.inf
		max_fitness = -math.inf

		for spec in self.population.species:
			for individual in spec.individuals:
				if individual.fitness < min_fitness:
					min_fitness = individual.fitness

				if individual.fitness > max_fitness:
					max_fitness = individual.fitness

		fitness_range = max(abs(max_fitness - min_fitness), 1.0)

		# compute adjusted fitness for every species
		adjusted_fitness_sum = 0

		for spec in self.population.species:
			mean_fitness = sum([individual.fitness for individual in spec.individuals]) / len(spec.individuals)
			spec.adjusted_fitness = (mean_fitness - min_fitness) / fitness_range
			adjusted_fitness_sum += spec.adjusted_fitness

		# compute num children
		total_spawn = 0

		# todo: weird formula
		for spec in self.population.species:
			adjusted_fitness = spec.adjusted_fitness
			size = len(spec.individuals)

			s = max(config.elitism, adjusted_fitness / adjusted_fitness_sum * config.pop_size)

			d = (s - size) * 0.5
			c = int(round(d))

			spawn = size
			if abs(c) > 0:
				spawn += c
			elif d > 0:
				spawn += 1
			elif d < 0:
				spawn -= 1

			spec.num_children = spawn
			total_spawn += spec.num_children

		norm = config.pop_size / total_spawn

		for spec in self.population.species:
			spec.num_children = max(config.elitism, round(spec.num_children * norm))

		# reproduce
		generation_new_nodes = {}
		generation_new_connections = {}

		children = []
		for spec in self.population.species:
			size = len(spec.individuals)

			# elitism
			num_elites = min(config.elitism, size)
			for i in range(num_elites):
				child = spec.individuals[i].duplicate()
				children.append(child)
				spec.num_children -= 1

			# survival threshold
			num_surviving = max(2, math.ceil(config.survival_threshold * size))
			spec.individuals = spec.individuals[:num_surviving]

			while spec.num_children > 0:
				# randomly select two parents
				parent1 = random.choice(spec.individuals)
				parent2 = random.choice(spec.individuals)

				# crossover or duplicate
				if parent1 == parent2:
					child = parent1.duplicate()
				else:
					# todo: check
					child = crossover([parent1, parent2])

				# mutate
				# todo: check
				child.mutate(generation_new_nodes, generation_new_connections)

				children.append(child)
				spec.num_children -= 1

			spec.reset()

		self.population.individuals = children

		# speciate and remove empty species
		self.population.speciate()

		return best_individual


class tsNEAT(AbstractNEAT):
	def epoch(self):
		self.population.speciate()

		best_individual = self.population.evaluate_fitness(self.evaluate)
		self.stats.update_generation(self.population)

		self.population.adjust_fitness()
		self.population.breed_new_generation_by_tournament_selection()

		return best_individual
