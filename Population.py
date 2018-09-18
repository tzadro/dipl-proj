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

		for spec in self.species:
			if spec.num_children == 0:
				continue

			if len(spec.individuals) == 0:
				self.species.remove(spec)
				continue

			spec.sort_individuals()

			# first add best one
			children = children + [spec.individuals[-1]]

			if spec.num_children == 1:
				continue

			# todo: do we need to remove worst?
			spec.remove_worst()

			# todo: duplicate code!
			if len(spec.individuals) == 0:
				self.species.remove(spec)
				continue

			children = children + [spec.breed_child(generation_innovations) for _ in range(spec.num_children - 1)]

			# todo: move?
			spec.clear()

		self.individuals = children
		self.max_fitness = 0
