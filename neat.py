from config import config
from population import Population
from species import Species
import utility
import math
import random


class NEAT:
	def __init__(self, evaluate):
		self.evaluate = evaluate
		self.population = Population()

	def epoch(self):
		self.population.speciate()

		best_individual, best_fitness, avg_fitness = self.population.evaluate_fitness(self.evaluate)

		self.population.adjust_fitness()
		self.population.assign_num_children()

		self.population.remove_worst()
		self.population.breed_new_generation()

		return best_individual, best_fitness, avg_fitness

	def reset(self):
		self.population = Population()


class tsNEAT:
	def __init__(self, evaluate):
		self.evaluate = evaluate
		self.population = Population()

	def epoch(self):
		self.population.speciate()

		best_individual, best_fitness, avg_fitness = self.population.evaluate_fitness(self.evaluate)

		self.population.adjust_fitness()
		self.population.breed_new_generation_by_tournament_selection()

		return best_individual, best_fitness, avg_fitness

	def reset(self):
		self.population = Population()


class StanleyNEAT:
	def __init__(self, evaluate):
		self.evaluate = evaluate
		self.population = Population()
		self.generation = 0

	def epoch(self):
		# evaluate
		if config.verbose: print("\tEvaluate")
		avg_fitness = 0
		for individual in self.population.individuals:
			individual.fitness = self.evaluate(individual.connections.values())
			avg_fitness += individual.fitness
		avg_fitness /= len(self.population.individuals)

		# speciate
		if config.verbose: print("\tSpeciate")
		for individual in self.population.individuals:
			placed = False

			for spec in self.population.species:
				dist_from_repr = utility.distance(individual, spec.representative)

				if dist_from_repr <= config.compatibility_threshold:
					spec.add(individual)
					placed = True
					break

			if not placed:
				new_spec = Species(individual)
				self.population.species.append(new_spec)

		# remove empty species
		self.population.species = [spec for spec in self.population.species if len(spec.individuals) > 0]

		# sort species by max unadjusted fitness
		if config.verbose: print("\tSort")

		def key(element):
			return -element.fitness

		for spec in self.population.species:
			spec.individuals.sort(key=key)

		def key(element):
			return -element.individuals[0].fitness

		self.population.species.sort(key=key)

		# save best individual for later
		best_individual = self.population.species[0].individuals[0]

		# flag the lowest performing species over age 20 every 30 generations
		if config.verbose: print("\tSpecies stagnation")
		if self.generation % 30 == 0:
			for spec in reversed(self.population.species):
				if spec.age >= 20:
					spec.obliterate = True
					break

		# penalize old, boost young, adjust fitness and mark for death
		if config.verbose: print("\tAdjust fitness")
		for spec in self.population.species:
			spec.stanley_adjust_fitness()

		# compute overall average fitness
		if config.verbose: print("\tAvg adjusted fitness")
		total_adjusted_fitness = sum([individual.adjusted_fitness for individual in self.population.individuals])
		num_individuals = len(self.population.individuals)
		overall_average = total_adjusted_fitness / num_individuals

		# compute expected number of offsprings for each individual organism
		if config.verbose: print("\tExpected num children")
		for individual in self.population.individuals:
			individual.expected_num_offsprings = individual.adjusted_fitness / overall_average

		# sum expected number of offsprings for every species
		if config.verbose: print("\tNum children per species")
		total_expected = 0
		for spec in self.population.species:
			spec.num_children = math.floor(sum([individual.expected_num_offsprings for individual in spec.individuals]))
			total_expected += spec.num_children

		# distribute children from loss in decimal points to random species by tournament selection
		if config.verbose: print("\tTotal expected", total_expected)
		while total_expected < config.pop_size:
			spec = random.choice(self.population.species)
			spec.num_children += 1
			total_expected += 1

		# check for population-level stagnation
		if config.verbose: print("\tPopulation stagnation")
		if best_individual.fitness > self.population.max_fitness:
			self.population.max_fitness = best_individual.fitness
			self.population.num_generations_before_last_improvement = 0
		else:
			self.population.num_generations_before_last_improvement += 1

		# if there is stagnation allow only first two species to reproduce
		if self.population.num_generations_before_last_improvement > config.max_num_generations_before_population_improvement:
			if config.verbose: print("\tStagnation detected")
			self.population.num_generations_before_last_improvement = 0

			if len(self.population.species) == 1:
				spec = self.population.species[0]
				spec.num_children = config.pop_size
				spec.num_generations_before_last_improvement = 0
			else:
				for spec in self.population.species[:2]:
					# assumes pop_size will always be even number
					spec.num_children = math.floor(config.pop_size / 2)
					spec.num_generations_before_last_improvement = 0
				for spec in self.population.species[2:]:
					spec.num_children = 0

		# delete all individuals marked for death
		if config.verbose: print("\tDelete all marked for death")
		for spec in self.population.species:
			spec.individuals = [individual for individual in spec.individuals if not individual.eliminate]

		# reproduce
		if config.verbose: print("\tReproduce")
		generation_new_nodes = {}
		generation_new_connections = {}

		children = []
		for spec in self.population.species:
			if config.verbose: print("\t\tSpec", self.population.species.index(spec))
			children += spec.stanley_reproduce(generation_new_nodes, generation_new_connections)

		self.population.individuals = children

		# end
		if config.verbose: print("\tEnd epoch")
		self.generation += 1
		return best_individual, best_individual.fitness, avg_fitness

	def reset(self):
		self.population = Population()
		self.generation = 0
