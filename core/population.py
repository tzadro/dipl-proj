from core.config import config
from core.individual import Individual, crossover
from core.species import Species
from core import utility
import math
import numpy as np


class Population:
	def __init__(self):
		self.individuals = [Individual() for _ in range(config.pop_size)]
		self.species = []
		self.next_species_key = 0
		self.speciate()

	def evaluate_fitness(self, evaluate):
		best_individual = None

		for individual in self.individuals:
			individual.fitness = evaluate(individual)

			if not best_individual or individual.fitness > best_individual.fitness:
				best_individual = individual

		return best_individual

	# from best to worst
	def sort(self):
		# sort individuals in every species
		for spec in self.species:
			spec.sort()

		# sort species by max fitness
		self.species.sort(key=lambda x: -x.individuals[0].fitness)

	def remove_stagnant_species(self):
		for spec in reversed(self.species):
			if len(self.species) <= config.min_num_species:
				break

			best_fitness = spec.individuals[0].fitness
			if best_fitness > spec.max_fitness_ever:
				spec.num_generations_before_last_improvement = 0
				spec.max_fitness_ever = best_fitness
			else:
				spec.num_generations_before_last_improvement += 1

				if spec.num_generations_before_last_improvement > config.max_num_generations_before_species_improvement:
					self.species.remove(spec)

	def adjust_species_fitness(self):
		# find min and max for normalization
		min_fitness = math.inf
		max_fitness = -math.inf

		for spec in self.species:
			for individual in spec.individuals:
				if individual.fitness < min_fitness:
					min_fitness = individual.fitness

				if individual.fitness > max_fitness:
					max_fitness = individual.fitness

		fitness_range = max(abs(max_fitness - min_fitness), 1.0)

		# compute adjusted fitness for every species
		for spec in self.species:
			mean_fitness = sum([individual.fitness for individual in spec.individuals]) / len(spec.individuals)
			spec.adjusted_fitness = (mean_fitness - min_fitness) / fitness_range

	def assign_num_children(self):
		total_spawn = 0
		adjusted_fitness_sum = sum([spec.adjusted_fitness for spec in self.species])

		# todo: weird formula
		for spec in self.species:
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

		for spec in self.species:
			spec.num_children = max(config.elitism, round(spec.num_children * norm))

		if config.elitism == 0:
			self.species = [spec for spec in self.species if spec.num_children > 0]

	def reproduce(self):
		# track new innovations in a generation to prevent giving same structural changes different innovation numbers
		generation_new_nodes = {}
		generation_new_connections = {}

		children = []
		for spec in self.species:
			size = len(spec.individuals)

			# elitism
			num_elites = min(config.elitism, size)
			for i in range(num_elites):
				child = spec.individuals[i].duplicate()
				children.append(child)
				spec.num_children -= 1

			# survival threshold
			num_surviving = max(2, math.ceil(config.survival_threshold * size))
			spec.trim_to(num_surviving)

			while spec.num_children > 0:
				# randomly select two parents
				parent1, parent2 = spec.random_select(2, True)

				# crossover or duplicate
				if parent1 == parent2:
					child = parent1.duplicate()
				else:
					child = crossover([parent1, parent2])

				# mutate
				child.mutate(generation_new_nodes, generation_new_connections)

				children.append(child)
				spec.num_children -= 1

			spec.reset()

		self.individuals = children

	def speciate(self):
		for individual in self.individuals:
			placed = False

			for spec in self.species:
				dist_from_repr = utility.distance(individual, spec.representative)

				if dist_from_repr <= config.compatibility_threshold:
					spec.add(individual)
					placed = True
					break

			if not placed:
				new_spec = Species(self.next_species_key, individual)
				self.species.append(new_spec)
				self.next_species_key += 1

		self.species = [spec for spec in self.species if len(spec.individuals) > 0]

	def adjust_compatibility_threshold(self):
		num_species = len(self.species)

		if num_species > config.desired_num_species:
			delta = config.ct_step
		elif num_species < config.desired_num_species:
			delta = -config.ct_step

		new_value = config.compatibility_threshold + delta
		config.compatibility_threshold = np.clip(new_value, config.ct_min_val, config.ct_max_val)
