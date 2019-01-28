from core.config import config
from core.individual import Individual
from core.species import Species
from core.interface import log
from core import utility
import math
import numpy as np


class Population:
	def __init__(self):
		self.species = []
		self.next_species_key = 0

		individuals = [Individual() for _ in range(config.pop_size)]
		self.speciate(individuals)

	def evaluate_fitness(self, evaluate):
		best_individual = None

		for spec in self.species:
			for individual in spec.individuals:
				individual.fitness = evaluate(individual)

				if not best_individual or individual.fitness > best_individual.fitness:
					best_individual = individual

		return best_individual

	# from best to worst, also sorts species
	def sort(self):
		# sort individuals in every species from best to worst
		for spec in self.species:
			spec.sort()

		# sort species by max fitness
		self.species.sort(key=lambda x: -x.individuals[0].fitness)

	# assumes species are sorted
	def remove_stagnant_species(self):
		# iterate through all species from worst to best
		for spec in reversed(self.species):
			# stop removing stagnant species if number of species is less than or equal to minimum number of species
			if len(self.species) <= config.min_num_species:
				break

			# if species improved reset counter, else increment it
			best_fitness = spec.individuals[0].fitness
			if best_fitness > spec.max_fitness_ever:
				spec.num_gens_before_last_improv = 0
				spec.max_fitness_ever = best_fitness
			else:
				spec.num_gens_before_last_improv += 1

				# if species hasn't improved in a long time remove it from the population
				if spec.num_gens_before_last_improv > config.max_num_gens_before_spec_improv:
					log('\t\tRemoving stagnant species {:d} after {:d} generations without improvement'.format(spec.key, spec.num_gens_before_last_improv))
					self.species.remove(spec)

	def adjust_species_fitness(self):
		# find minimum and maximum fitness for normalization
		min_fitness = math.inf
		max_fitness = -math.inf
		for spec in self.species:
			for individual in spec.individuals:
				if individual.fitness < min_fitness:
					min_fitness = individual.fitness

				if individual.fitness > max_fitness:
					max_fitness = individual.fitness

		# define fitness range
		fitness_range = max(abs(max_fitness - min_fitness), 1.0)

		for spec in self.species:
			# calculate species mean fitness
			mean_fitness = sum([individual.fitness for individual in spec.individuals]) / len(spec.individuals)
			# set adjusted fitness as normalized mean fitness
			spec.adjusted_fitness = (mean_fitness - min_fitness) / fitness_range

	def assign_num_children(self):
		total_spawn = 0
		adjusted_fitness_sum = sum([spec.adjusted_fitness for spec in self.species])

		for spec in self.species:
			adjusted_fitness = spec.adjusted_fitness
			size = len(spec.individuals)

			# calculate potential number of children proportionally to species adjusted fitness
			potential_size = max(config.elitism, adjusted_fitness / adjusted_fitness_sum * config.pop_size)
			# calculate difference between current size and potential number of children
			size_delta = potential_size - size
			# set number of children somewhere between current and potential size depending on smoothing coefficient
			num_children = size + round(size_delta * config.spawn_smooth_coef)

			spec.num_children = num_children
			total_spawn += num_children

		# calculate coefficient with which we will normalize all number of children
		norm = config.pop_size / total_spawn

		for spec in self.species:
			# by normalizing we assure population size will always be as close to defined as possible
			spec.num_children = max(config.elitism, round(spec.num_children * norm))

		# if there is no elitism in species it is possible some have 0 children assigned, remove those species if so
		if config.elitism == 0:
			log('\t\tRemoving species {:d} because no children were assigned'.format(self.spec.key))
			self.species = [spec for spec in self.species if spec.num_children > 0]

	def reproduce(self):
		# track new innovations in a generation to prevent giving same structural changes different innovation numbers
		generation_new_nodes = {}
		generation_new_connections = {}

		children = []
		for spec in self.species:
			children += spec.reproduce(generation_new_nodes, generation_new_connections)

		return children

	def speciate(self, children):
		# assign every individual to a species
		for individual in children:
			placed = False

			Es = []
			Ds = []
			weight_diffs = []

			# first see if individual is compatible with any existing species
			for spec in self.species:
				dist_from_repr, E, D, weight_diff = utility.distance(individual, spec.representative)

				Es.append(E)
				Ds.append(D)
				weight_diffs.append(weight_diff)

				# if distance from species representative is smaller than threshold add individual to that species
				if dist_from_repr <= config.compatibility_threshold:
					spec.add(individual)
					placed = True
					break

			# if individual is not placed to any existing species create new species and set it as representative
			if not placed:
				new_spec = Species(self.next_species_key, individual)
				self.species.append(new_spec)
				self.next_species_key += 1

		# log empty species
		for spec in self.species:
			if len(spec.individuals) == 0:
				log('\t\tSpecies {:d} is empty after speciation'.format(spec.key))

		# delete any species that is empty after speciation
		self.species = [spec for spec in self.species if len(spec.individuals) > 0]

		return Es, Ds, weight_diffs

	def adjust_compatibility_threshold(self):
		num_species = len(self.species)

		# calculate compatibility threshold modifier depending on current number of species
		delta = 0
		if num_species > config.max_desired_num_species:
			delta = config.ct_step
		elif num_species < config.min_desired_num_species:
			delta = -config.ct_step

		# adjust current value with calculated modifier but make sure it doesn't go over defined boundaries
		new_value = config.compatibility_threshold + delta
		config.compatibility_threshold = np.clip(new_value, config.ct_min_val, config.ct_max_val)
