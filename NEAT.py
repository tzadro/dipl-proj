from Config import config
from Population import Population
import Interface


class NEAT:
	def __init__(self, env):
		self.env = env
		self.env.evaluations = 0
		self.env.solved = False

		self.best_fitnesses = []
		self.avg_fitnesses = []

		self.population = Population()

	def run(self):
		for i in range(config.num_iter):
			self.population.speciate()

			visualize = i % config.visualize_every == 0
			best_fitness, avg_fitness = self.population.evaluate_fitness(self.env, visualize)
			self.best_fitnesses.append(best_fitness)
			self.avg_fitnesses.append(avg_fitness)

			self.population.adjust_fitness()
			self.population.assign_num_children()

			if config.verbose:
				Interface.verbose(i, self.population, best_fitness, avg_fitness)

			if self.env.solved:
				return self.env.evaluations

			self.population.remove_worst()
			self.population.breed_new_generation()

		return None

	def reset(self):
		self.env.evaluations = 0
		self.env.solved = False

		self.best_fitnesses = []
		self.avg_fitnesses = []

		self.population = Population()
