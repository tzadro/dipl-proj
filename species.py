from config import config
from individual import crossover
import random
import copy
import math
import numpy as np


class Species:
	def __init__(self, representative):
		self.representative = copy.deepcopy(representative)
		self.individuals = [representative]
		self.fitness = None
		self.max_fitness = -math.inf
		self.num_generations_before_last_improvement = None
		self.num_children = None

	def add(self, individual):
		self.individuals.append(individual)

	def adjust_fitness(self):
		self.fitness = 0

		num_individuals = len(self.individuals)
		for individual in self.individuals:
			individual.adjusted_fitness = individual.fitness / num_individuals
			self.fitness += individual.adjusted_fitness

		if self.fitness > self.max_fitness:
			self.max_fitness = self.fitness
			self.num_generations_before_last_improvement = 0
		else:
			self.num_generations_before_last_improvement += 1

	# from best to worst
	def sort(self):
		def key(element):
			return -element.adjusted_fitness

		self.individuals.sort(key=key)

	def breed_child(self, generation_new_nodes, generation_new_connections):
		if len(self.individuals) == 1 or random.random() < config.skip_crossover_probability:
			child = copy.deepcopy(self.select())
		else:
			child = crossover(self.select(2))

		child.mutate(generation_new_nodes, generation_new_connections)
		return child

	def select(self, size=None, replace=False):
		p = [individual.adjusted_fitness / self.fitness for individual in self.individuals]
		individual = np.random.choice(self.individuals, size, replace, p)
		return individual

	def trim_to(self, n=1):
		self.individuals = self.individuals[:n]

	def clear(self):
		random_individual = random.choice(self.individuals)
		self.representative = copy.deepcopy(random_individual)
		self.individuals = []
		self.fitness = None
		self.num_children = None
