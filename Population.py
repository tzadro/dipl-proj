from Config import config
from Individual import Individual
from Species import Species
import helperfunctions
import math


class Population:
	def __init__(self):
		self.individuals = [Individual() for _ in range(config.pop_size)]
		self.species = []

	def evaluate_fitness(self, env):
		best_fitness = -math.inf
		avg_fitness = 0

		for individual in self.individuals:
			fitness = individual.evaluate_fitness(env)
			best_fitness = max(best_fitness, fitness)
			avg_fitness += fitness

		avg_fitness /= len(self.individuals)
		return best_fitness, avg_fitness

	def speciate(self):
		for individual in self.individuals:
			placed = False

			for spec in self.species:
				dist_from_repr = helperfunctions.distance(individual, spec.representative)

				if dist_from_repr <= config.compatibility_threshold:
					spec.add(individual)
					placed = True
					break

			if not placed:
				new_spec = Species(individual)
				self.species.append(new_spec)

		for spec in self.species:
			if len(spec.individuals) == 0:
				self.species.remove(spec)

	def adjust_fitness(self):
		for spec in self.species:
			spec.adjust_fitness()
			spec.sort()

	def assign_num_children(self):
		sum_spec_fitness = sum([spec.fitness for spec in self.species])

		for spec in self.species:
			spec.num_children = math.floor(spec.fitness / sum_spec_fitness * config.pop_size)

			if spec.num_children == 0:
				self.species.remove(spec)

	def remove_worst(self):
		for spec in self.species:
			num_surviving = math.floor(len(spec.individuals) * config.survival_threshold) + 1
			spec.trim_to(num_surviving)

	def breed_new_generation(self):
		children = []
		generation_innovations = {}

		for spec in self.species:
			# todo: don't add best one for every species?
			children += [spec.individuals[0]] + [spec.breed_child(generation_innovations) for _ in range(spec.num_children - 1)]

			spec.clear()

		self.individuals = children
