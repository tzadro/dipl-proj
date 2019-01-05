from core.population import Population


class AbstractNEAT:
	def __init__(self, evaluate, stats):
		self.evaluate = evaluate
		self.stats = stats
		self.population = Population()

	def epoch(self):
		pass

	def reset(self):
		self.stats.reset_generation()
		self.population = Population()


class NEAT(AbstractNEAT):
	def epoch(self):
		# evaluate population
		best_individual = self.population.evaluate_fitness(self.evaluate)
		self.stats.update_generation(self.population)

		self.population.sort()
		self.population.remove_stagnant_species()

		# normalize
		self.population.adjust_species_fitness()

		# normaliz
		self.population.assign_num_children()

		# copy elites, remove worst and breed new generation
		self.population.reproduce()

		# speciate and remove empty species
		self.population.speciate()

		return best_individual
