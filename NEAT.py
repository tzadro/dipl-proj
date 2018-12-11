from Config import config
from Population import Population


class NEAT:
	def __init__(self, env):
		self.env = env
		self.population = Population()

	def run(self):
		for i in range(config.num_iter):
			self.epoch()

			if self.env.solved:
				return

	def epoch(self):
		self.population.speciate()

		best_individual, best_fitness, avg_fitness = self.population.evaluate_fitness(self.env)

		self.population.adjust_fitness()
		self.population.assign_num_children()

		self.population.remove_worst()
		self.population.breed_new_generation()

		return best_individual, best_fitness, avg_fitness

	def reset(self):
		self.env.solved = False
		self.population = Population()
