from config import config
from population import Population
from individual import crossover
from interface import log
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


class NewNEAT:
	def __init__(self, evaluate):
		self.evaluate = evaluate
		self.population = Population()

	def epoch(self):
		# evaluate population
		best_individual, best_individual.fitness, avg_fitness = self.population.evaluate_fitness(self.evaluate)

		# sort by max unadjusted fitness
		for spec in self.population.species:
			spec.individuals.sort(key=lambda x: -x.fitness)
		self.population.species.sort(key=lambda x: -x.individuals[0].fitness)

		# remove stagnant species
		for spec in reversed(self.population.species):
			if len(self.population.species) <= config.species_elitism:
				break

			best_fitness = spec.individuals[0].fitness
			if best_fitness > spec.max_fitness_ever:
				spec.num_generations_before_last_improvement = 0
				spec.max_fitness_ever = best_fitness
			else:
				spec.num_generations_before_last_improvement += 1

				if spec.num_generations_before_last_improvement > config.max_num_generations_before_species_improvement:
					self.population.species.remove(spec)

		# find min and max for normalization
		min_fitness = math.inf
		max_fitness = -math.inf

		for spec in self.population.species:
			for individual in spec.individuals:
				if individual.fitness < min_fitness:
					min_fitness = individual.fitness

				if individual.fitness > max_fitness:
					max_fitness = individual.fitness

		fitness_range = abs(max_fitness - min_fitness)

		# compute adjusted fitness for every species
		adjusted_fitness_sum = 0

		for spec in self.population.species:
			mean_fitness = sum([individual.fitness for individual in spec.individuals]) / len(spec.individuals)
			spec.adjusted_fitness = (mean_fitness - min_fitness) / fitness_range
			adjusted_fitness_sum += spec.adjusted_fitness

		# compute num children
		total_spawn = 0

		# todo: weird formula
		for spec in self.population.species:
			adjusted_fitness = spec.adjusted_fitness
			size = len(spec.individuals)

			s = max(config.elitism, adjusted_fitness / adjusted_fitness_sum * config.pop_size)

			d = (s - size) * 0.5
			c = int(round(d))

			spawn = size
			if abs(c) > 0:
				spawn += c
			elif d > 0:
				spawn += 1
			elif d < 0:
				spawn -= 1

			spec.num_children = spawn
			total_spawn += spec.num_children

		norm = config.pop_size / total_spawn

		for spec in self.population.species:
			spec.num_children = max(config.elitism, round(spec.num_children * norm))

		# reproduce
		generation_new_nodes = {}
		generation_new_connections = {}

		children = []
		for spec in self.population.species:
			size = len(spec.individuals)

			# elitism
			num_elites = min(config.elitism, size)
			for i in range(num_elites):
				child = spec.individuals[i].duplicate()
				children.append(child)
				spec.num_children -= 1

			# survival threshold
			num_surviving = max(2, math.ceil(config.survival_threshold * size))
			spec.individuals = spec.individuals[:num_surviving]

			while spec.num_children > 0:
				# randomly select two parents
				parent1 = random.choice(spec.individuals)
				parent2 = random.choice(spec.individuals)

				# crossover or duplicate
				if parent1 == parent2:
					child = parent1.duplicate()
				else:
					# todo: check
					child = crossover([parent1, parent2])

				# mutate
				# todo: check
				child.mutate(generation_new_nodes, generation_new_connections)

				children.append(child)
				spec.num_children -= 1

			spec.clear()

		self.population.individuals = children

		# speciate and remove empty species
		self.population.speciate()

		return best_individual, best_individual.fitness, avg_fitness

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
		log("\t\tEvaluate")
		avg_fitness = 0
		for individual in self.population.individuals:
			individual.fitness = self.evaluate(individual.connections.values())
			avg_fitness += individual.fitness
		avg_fitness /= len(self.population.individuals)

		# speciate and remove empty species
		log("\t\tSpeciate")
		self.population.speciate()

		# sort species by max unadjusted fitness
		log("\t\tSort")

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
		log("\t\tSpecies stagnation")
		if self.generation % 30 == 0:
			for spec in reversed(self.population.species):
				if spec.age >= 20:
					spec.obliterate = True
					break

		# penalize old, boost young, adjust fitness and mark for death
		log("\t\tAdjust fitness")
		for spec in self.population.species:
			spec.stanley_adjust_fitness()

		# compute overall average fitness
		log("\t\tAvg adjusted fitness")
		total_adjusted_fitness = sum([individual.adjusted_fitness for individual in self.population.individuals])
		num_individuals = len(self.population.individuals)
		overall_average = total_adjusted_fitness / num_individuals

		# compute expected number of offsprings for each individual organism
		log("\t\tExpected num children")
		for individual in self.population.individuals:
			individual.expected_num_offsprings = individual.adjusted_fitness / overall_average

		# sum expected number of offsprings for every species
		log("\t\tNum children per species")
		total_expected = 0
		for spec in self.population.species:
			spec.num_children = math.floor(sum([individual.expected_num_offsprings for individual in spec.individuals]))
			total_expected += spec.num_children

		# distribute children from loss in decimal points to random species by tournament selection
		log("\t\tTotal expected {:d}".format(total_expected))
		while total_expected < config.pop_size:
			spec = random.choice(self.population.species)
			spec.num_children += 1
			total_expected += 1

		# check for population-level stagnation
		log("\t\tPopulation stagnation")
		if best_individual.fitness > self.population.max_fitness:
			self.population.max_fitness = best_individual.fitness
			self.population.num_generations_before_last_improvement = 0
		else:
			self.population.num_generations_before_last_improvement += 1

		# if there is stagnation allow only first two species to reproduce
		if self.population.num_generations_before_last_improvement > config.max_num_generations_before_population_improvement:
			log("\t\tStagnation detected")
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
		log("\t\tDelete all marked for death")
		for spec in self.population.species:
			spec.individuals = [individual for individual in spec.individuals if not individual.eliminate]

		# reproduce
		log("\t\tReproduce")
		generation_new_nodes = {}
		generation_new_connections = {}

		children = []
		for spec in self.population.species:
			log("\t\t\tSpec {:d}".format(self.population.species.index(spec)))
			children += spec.stanley_reproduce(generation_new_nodes, generation_new_connections)

		self.population.individuals = children

		# end
		log("\t\tEnd epoch")
		self.generation += 1
		return best_individual, best_individual.fitness, avg_fitness

	def reset(self):
		self.population = Population()
		self.generation = 0
