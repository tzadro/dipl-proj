from config import config
from individual import crossover
from interface import log
import random
import math
import numpy as np


class Species:
	def __init__(self, representative):
		self.representative = representative.duplicate()
		self.individuals = [representative]
		self.adjusted_fitness = None
		self.max_fitness_ever = -math.inf
		self.num_generations_before_last_improvement = 0
		self.num_children = None

		# used only for stanley neat
		self.obliterate = False
		self.age = 0

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

	def clear(self):
		random_individual = self.random_select()
		self.representative = random_individual.duplicate()
		self.individuals = []
		self.adjusted_fitness = None
		self.num_children = None
		self.age += 1

	def stanley_adjust_fitness(self):
		num_individuals = len(self.individuals)

		# true if species hasn't improved in enough time
		age_debt = self.num_generations_before_last_improvement > config.max_num_generations_before_species_improvement

		for individual in self.individuals:
			# don't allow negative fitness
			assert individual.fitness > 0, 'Individual\'s fitness must be positive'

			# adjusted fitness is original fitness shared with the species
			individual.adjusted_fitness = individual.fitness / num_individuals

			# penalize stagnating species
			if age_debt or self.obliterate:
				individual.adjusted_fitness *= config.stagnation_penalization

			# boost young species
			if self.age <= config.youth_threshold:
				individual.adjusted_fitness *= config.youth_boost

		# update age of last improvement, assumes individuals are already sorted
		best_individual = self.individuals[0]
		if best_individual.fitness > self.max_fitness_ever:
			self.max_fitness_ever = best_individual.fitness
			self.num_generations_before_last_improvement = 0
		else:
			self.num_generations_before_last_improvement += 1

		# decide how many get to reproduce
		num_parents = math.floor(num_individuals * config.survival_threshold) + 1

		# mark for death those who are ranked too low to be parents
		for individual in self.individuals[num_parents:]:
			individual.eliminate = True

	def stanley_reproduce(self, generation_new_nodes, generation_new_connections):
		log("\t\t\t\tNum children {:d}".format(self.num_children))
		if self.num_children == 0:
			return []

		num_individuals = len(self.individuals)

		if num_individuals >= config.min_num_individuals_for_elitism:
			log("\t\t\t\tBest individual copied (Elitism)")
			children = [self.individuals[0].duplicate()]
			self.num_children -= 1
		else:
			children = []

		while self.num_children > 0:
			# todo: check mutations
			if num_individuals == 1 or random.random() < config.skip_crossover_probability:
				child = self.random_select().duplicate()
				child.stanley_mutate(generation_new_nodes, generation_new_connections)
			else:
				# todo: add interspecies mating rate
				# todo: check crossovers
				child = crossover(self.random_select(2))

				if random.random() > config.mate_only_probability:
					child.stanley_mutate(generation_new_nodes, generation_new_connections)

			children.append(child)
			self.num_children -= 1

		self.clear()
		return children
