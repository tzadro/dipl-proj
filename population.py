from config import config
from individual import Individual
from species import Species
import utility
import math
import copy


class Population:
	def __init__(self):
		self.individuals = [Individual() for _ in range(config.pop_size)]
		self.species = []
		self.max_fitness = -math.inf
		self.num_generations_before_last_improvement = None

	def evaluate_fitness(self, evaluate):
		best_individual = None
		avg_fitness = 0

		for individual in self.individuals:
			individual.fitness = evaluate(individual.connections.values())
			avg_fitness += individual.fitness

			if not best_individual or individual.fitness > best_individual.fitness:
				best_individual = individual

		avg_fitness /= len(self.individuals)
		return best_individual, best_individual.fitness, avg_fitness

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
				new_spec = Species(individual)
				self.species.append(new_spec)

		self.species = [spec for spec in self.species if len(spec.individuals) > 0]

	def adjust_fitness(self):
		for spec in self.species:
			spec.adjust_fitness()
			spec.sort()
		self.sort()

		if len(self.species) > 1:
			for spec in self.species[1:]:
				if spec.num_generations_before_last_improvement <= config.max_num_generations_before_species_improvement:
					self.species.remove(spec)

		# todo: do this but sort species by species fitness, not by best fitted individual in the species
		if self.species[0].individuals[0].fitness > self.max_fitness:
			self.max_fitness = self.species[0].individuals[0].fitness
			self.num_generations_before_last_improvement = 0
		else:
			self.num_generations_before_last_improvement += 1

			if len(self.species) > 2:
				if self.num_generations_before_last_improvement > config.max_num_generations_before_population_improvement:
					self.species = self.species[:2]

					self.max_fitness = -math.inf
					self.num_generations_before_last_improvement = 0

	# from best to worst
	def sort(self):
		def key(element):
			return -element.individuals[0].fitness

		self.species.sort(key=key)

	def assign_num_children(self):
		sum_spec_fitness = sum([spec.fitness for spec in self.species])

		for spec in self.species:
			spec.num_children = math.floor(spec.fitness / sum_spec_fitness * (config.pop_size - len(self.species))) + 1

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
				children += [copy.deepcopy(spec.individuals[0])]
				spec.num_children -= 1

			children += [spec.breed_child(generation_new_nodes, generation_new_connections) for _ in range(spec.num_children)]

			spec.clear()

		self.individuals = children
