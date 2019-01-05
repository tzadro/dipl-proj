from core.config import config
from core.individual import crossover
import random
import math
import numpy as np


class Species:
	def __init__(self, key, representative):
		self.key = key
		self.representative = representative.duplicate()
		self.individuals = [representative]
		self.adjusted_fitness = None
		self.max_fitness_ever = -math.inf
		self.num_generations_before_last_improvement = 0
		self.num_children = None

	def add(self, individual):
		self.individuals.append(individual)

	def adjust_fitness(self):
		self.adjusted_fitness = 0

		num_individuals = len(self.individuals)
		for individual in self.individuals:
			individual.adjusted_fitness = individual.fitness / num_individuals
			self.adjusted_fitness += individual.adjusted_fitness

		if self.adjusted_fitness > self.max_fitness_ever:
			self.max_fitness_ever = self.adjusted_fitness
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
			child = self.roulette_select().duplicate()
		else:
			child = crossover(self.roulette_select(2))

		child.mutate(generation_new_nodes, generation_new_connections)
		return child

	def breed_child_by_tournament_selection(self, generation_new_nodes, generation_new_connections):
		def key(element):
			return -element.adjusted_fitness

		if len(self.individuals) == 1 or random.random() < config.skip_crossover_probability:
			child = self.random_select().duplicate()
		elif len(self.individuals) == 2:
			child = crossover(self.random_select(2))
		else:
			tournament = self.random_select(3)
			tournament.sort(key=key)
			child = crossover(tournament[:2])

		child.mutate(generation_new_nodes, generation_new_connections)
		return child

	# roulette wheel selection
	def roulette_select(self, size=None, replace=False):
		fitness_sum = sum([individual.fitness for individual in self.individuals])
		p = [individual.fitness / fitness_sum for individual in self.individuals]
		return np.random.choice(self.individuals, size, replace, p)

	# random selection
	def random_select(self, size=1):
		if size == 1:
			return random.choice(self.individuals)
		else:
			return random.sample(self.individuals, size)

	def trim_to(self, n=1):
		self.individuals = self.individuals[:n]

	def reset(self):
		random_individual = self.random_select()
		self.representative = random_individual.duplicate()
		self.individuals = []
		self.adjusted_fitness = None
		self.num_children = None
