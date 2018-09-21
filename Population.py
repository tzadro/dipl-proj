from Config import config
from Individual import Individual
from Species import Species
import helperfunctions
import math


class Population:
	def __init__(self):
		self.individuals = [Individual() for _ in range(config.pop_size)]
		self.species = []
		self.max_fitness = -math.inf

	def evaluate_fitness(self, env):
		for individual in self.individuals:
			self.max_fitness = max(self.max_fitness, individual.evaluate_fitness(env))

		return self.max_fitness

	def speciate(self):
		for individual in self.individuals:
			placed = False

			for spec in self.species:
				distance_from_representative = helperfunctions.distance(individual, spec.representative)

				if distance_from_representative <= config.compatibility_threshold:
					spec.add(individual)
					placed = True
					break

			if not placed:
				self.species.append(Species(individual))

	def adjust_fitness_and_calculate_num_children(self):
		sum_species_fitness = 0

		for spec in self.species:
			spec.adjust_fitness()
			sum_species_fitness = sum_species_fitness + spec.species_fitness

		for spec in self.species:
			spec.num_children = math.floor(spec.species_fitness / sum_species_fitness * config.pop_size)

	def breed_new_generation(self):
		self.adjust_fitness_and_calculate_num_children()

		children = []
		generation_innovations = {}

		for i, spec in enumerate(self.species):
			print('\tSpecies: {:d}, num_individuals: {:d}, num_children: {:d}'.format(i, len(spec.individuals), spec.num_children))

			if spec.num_children == 0 or len(spec.individuals) == 0:
				self.species.remove(spec)
				continue

			spec.sort()
			print('\t\tSpecies: {:d}, best fitness: {:.2f}'.format(i, spec.individuals[0].fitness))

			# first add best one
			children = children + [spec.individuals[0]]

			num_surviving = math.floor(len(spec.individuals) * config.survival_threshold) + 1
			spec.trim_to(num_surviving)

			children = children + [spec.breed_child(generation_innovations) for _ in range(spec.num_children - 1)]

			spec.clear()

		self.individuals = children
		self.max_fitness = -math.inf
