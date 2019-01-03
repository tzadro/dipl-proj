from config import config
from individual import Individual
from species import Species
import utility
import math
import random


class Population:
	def __init__(self):
		self.individuals = [Individual() for _ in range(config.pop_size)]
		self.species = []
		self.next_species_key = 0
		self.speciate()

		self.max_fitness_ever = -math.inf
		self.num_generations_before_last_improvement = None

	def evaluate_fitness(self, evaluate):
		best_individual = None

		for individual in self.individuals:
			individual.fitness = evaluate(individual)

			if not best_individual or individual.fitness > best_individual.fitness:
				best_individual = individual

		return best_individual

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

	def adjust_fitness(self):
		for spec in self.species:
			spec.adjust_fitness()
			spec.sort()
		self.sort()

		# todo: check if ordering is ok
		if len(self.species) > 1:
			self.species = [spec for spec in self.species if spec.num_generations_before_last_improvement <= config.max_num_generations_before_species_improvement or spec == self.species[0]]
		if len(self.species) > 2:
			if self.species[0].individuals[0].fitness > self.max_fitness_ever:
				self.max_fitness_ever = self.species[0].individuals[0].fitness
				self.num_generations_before_last_improvement = 0
			else:
				self.num_generations_before_last_improvement += 1
				if self.num_generations_before_last_improvement > config.max_num_generations_before_population_improvement:
					self.species = self.species[:2]

					self.max_fitness_ever = -math.inf
					self.num_generations_before_last_improvement = 0

	# from best to worst
	def sort(self):
		def key(element):
			return -element.individuals[0].fitness

		# todo: sort species by species fitness, not by best fitted individual in the species
		self.species.sort(key=key)

	def assign_num_children(self):
		sum_spec_fitness = sum([spec.adjusted_fitness for spec in self.species])

		for spec in self.species:
			spec.num_children = math.floor(spec.adjusted_fitness / sum_spec_fitness * (config.pop_size - len(self.species))) + 1

		# todo: this won't ever trigger, either don't give every species at least one spot or remove this
		self.species = [spec for spec in self.species if spec.num_children > 0]

	def remove_worst(self):
		for spec in self.species:
			num_surviving = math.floor(len(spec.individuals) * config.survival_threshold) + 1
			spec.trim_to(num_surviving)

	def breed_new_generation(self):
		children = []

		# track new innovations in a generation to prevent giving same structural changes different innovation numbers
		generation_new_nodes = {}
		generation_new_connections = {}

		for spec in self.species:
			if len(spec.individuals) > config.min_num_individuals_for_elitism:
				children += [spec.individuals[0].duplicate()]
				spec.num_children -= 1

			children += [spec.breed_child(generation_new_nodes, generation_new_connections) for _ in range(spec.num_children)]

			spec.reset()

		self.individuals = children

	def breed_new_generation_by_tournament_selection(self):
		children = []

		# track new innovations in a generation to prevent giving same structural changes different innovation numbers
		generation_new_nodes = {}
		generation_new_connections = {}

		for spec in self.species:
			children += [spec.individuals[0].duplicate()]

		while len(children) < config.pop_size:
			if len(self.species) == 1:
				spec = self.species[0]
			else:
				specs = random.sample(self.species, 2)
				spec = specs[0] if specs[0].fitness > specs[1].fitness else specs[1]

			children += [spec.breed_child_by_tournament_selection(generation_new_nodes, generation_new_connections)]

		for spec in self.species:
			spec.reset()

		self.individuals = children
