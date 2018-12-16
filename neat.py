from population import Population


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
