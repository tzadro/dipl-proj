from Config import config
from Individual import Individual
from Species import Species
from Interface import Interface
import helperfunctions
import math


class Population:
	def __init__(self):
		self.individuals = [Individual() for _ in range(config.pop_size)]
		self.species = []
		self.interface = Interface()

	def evaluate_fitness(self, env):
		best_individual = None
		avg_fitness = 0

		for individual in self.individuals:
			avg_fitness += individual.evaluate_fitness(env)

			if best_individual is None or individual.fitness > best_individual.fitness:
				best_individual = individual

		if config.visualize_best_networks:
			self.interface.visualize_network(best_individual.connections, best_individual.nodes)

		avg_fitness /= len(self.individuals)
		return best_individual.fitness, avg_fitness

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

		self.species = [spec for spec in self.species if len(spec.individuals) > 0]

	def adjust_fitness(self):
		for spec in self.species:
			spec.adjust_fitness()
			spec.sort()

		if len(self.species) > 1:
			self.species = [spec for spec in self.species if spec.num_generations_before_last_improvement <= config.max_num_generations_before_improvement]

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
		generation_innovations = {}

		for spec in self.species:
			if len(spec.individuals) > config.min_num_individuals_for_elitism:
				children += [spec.individuals[0]]
				spec.num_children -= 1

			children += [spec.breed_child(generation_innovations) for _ in range(spec.num_children)]

			spec.clear()

		self.individuals = children

		for individual in self.individuals:
			self.interface.update_node_positions(individual.connections, individual.nodes)
