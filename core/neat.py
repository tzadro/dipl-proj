from core.config import config
from core.population import Population
from core.interface import log


class AbstractNEAT:
	def __init__(self, evaluate, stats):
		self.evaluate = evaluate
		self.stats = stats
		self.population = Population()

		self.starting_compatibility_threshold = config.compatibility_threshold

	def epoch(self):
		pass

	def reset(self):
		self.stats.reset_generation()
		self.population = Population()

		config.compatibility_threshold = self.starting_compatibility_threshold


class NEAT(AbstractNEAT):
	def epoch(self):
		# evaluate population and track stats
		best_individual = self.population.evaluate_fitness(self.evaluate)
		self.stats.update_fitnesses(self.population.individuals)
		self.stats.update_structures(self.population.individuals)
		log('\tBest fitness: {:.2f}, Average fitness: {:.2f}'.format(self.stats.best_fitnesses[-1], self.stats.avg_fitnesses[-1]))
		log('\tAverage num hidden nodes: {:.2f}, Average num connections: {:.2f}'.format(self.stats.avg_num_hidden_nodes[-1], self.stats.avg_num_connections[-1]))

		# sort individuals inside each species and then sort species by their max fitness
		self.population.sort()
		log('\tBest individual is in species {:d}'.format(self.population.species[0].key))

		# remove species that showed no progress in recent generations
		self.population.remove_stagnant_species()

		# normalize adjusted fitness range and assign adjusted fitness to each species
		self.population.adjust_species_fitness()

		# assign children according to adjusted fitness and remove species with no children
		self.population.assign_num_children()

		# copy elites if enabled, remove worst in each species and breed new generation
		self.population.reproduce()

		# speciate new individuals and remove empty species
		Es, Ds, weight_diffs = self.population.speciate()
		self.stats.update_species(self.population.species, self.population.next_species_key)
		self.stats.update_distances(Es, Ds, weight_diffs)
		log('\tNum species is {:d}'.format(len(self.population.species)))
		log('\tAverage distance is {:.2f}'.format(sum(weight_diffs) / len(weight_diffs)))

		# change compatibility threshold if number of species becomes too big or too small
		if config.adjust_compatibility_threshold:
			self.population.adjust_compatibility_threshold()
			log('\tCompatibility threshold is {:.2f}'.format(config.compatibility_threshold))

		return best_individual
