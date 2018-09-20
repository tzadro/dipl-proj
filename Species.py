from Config import config
from Individual import crossover
"""import math"""
import random
import copy
import numpy as np


class Species:
	def __init__(self, representative):
		self.representative = copy.deepcopy(representative)
		"""self.current_closest = (representative, 0)"""
		self.individuals = [representative]
		self.species_fitness = 0
		self.num_children = None

	def add(self, individual):
		self.individuals.append(individual)

	# todo: what to do with this?
	"""
	def set_representative(self):
		for individual in self.individuals:
			distance_from_representative = helperfunctions.distance(individual, self.representative)
			if distance_from_representative < self.current_closest[1]:
				self.current_closest = (individual, distance_from_representative)

		self.representative = self.current_closest[0]
		self.current_closest = (None, math.inf)
	"""

	def adjust_fitness(self):
		for individual in self.individuals:
			individual.fitness = individual.fitness / len(self.individuals)
			self.species_fitness = self.species_fitness + individual.fitness

	def breed_child(self, generation_innovations):
		if random.random() < config.crossover_probability and len(self.individuals) > 1:
			child = crossover(self.select(2))
		else:
			child = copy.deepcopy(self.select())

		child.mutate(generation_innovations)
		return child

	def select(self, size=None, replace=False):
		fitness_sum = sum([individual.fitness for individual in self.individuals])
		p = [individual.fitness / fitness_sum for individual in self.individuals]
		return np.random.choice(self.individuals, size, replace, p)

	# from best to worst
	def sort(self):
		def key(element):
			return -element.fitness

		self.individuals.sort(key=key)

	def trim_to(self, n=1):
		self.individuals = self.individuals[:n]

	def clear(self):
		# todo: set random individual as representative?
		self.representative = copy.deepcopy(self.individuals[0])
		self.individuals = []
		self.species_fitness = 0
		self.num_children = None
