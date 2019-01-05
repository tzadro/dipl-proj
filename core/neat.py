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
		# evaluate population and track stats
		best_individual = self.population.evaluate_fitness(self.evaluate)
		self.stats.update_generation(self.population)

		# sort individuals inside each species and then sort species by their max fitness
		self.population.sort()

		# remove species that showed no progress in recent generations
		self.population.remove_stagnant_species()

		# normalize adjusted fitness range and assign adjusted fitness to each species
		self.population.adjust_species_fitness()

		# assign children according to adjusted fitness and remove species with no children
		self.population.assign_num_children()

		# copy elites if enabled, remove worst in each species and breed new generation
		self.population.reproduce()

		# speciate new individuals and remove empty species
		self.population.speciate()

		return best_individual
