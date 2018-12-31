from config import config
from multiprocessing import Pool


class DefaultEvaluator:
	def __init__(self, env):
		self.env = env

	def evaluate_all(self, individuals):
		best_individual = None
		avg_fitness = 0

		for individual in individuals:
			fitness, solved = self.env.evaluate(individual, config)
			individual.fitness = fitness
			avg_fitness += individual.fitness
			self.env.evaluations += 1

			if solved:
				self.env.solved = True

			if not best_individual or individual.fitness > best_individual.fitness:
				best_individual = individual

		avg_fitness /= len(individuals)
		return best_individual, best_individual.fitness, avg_fitness


class ParallelEvaluator:
	def __init__(self, env):
		self.env = env
		self.pool = Pool(processes=4)

	def evaluate_all(self, individuals):
		best_individual = None
		avg_fitness = 0

		jobs = []
		for individual in individuals:
			jobs.append(self.pool.apply_async(self.env.evaluate, (individual, config)))

		for job, individual in zip(jobs, individuals):
			fitness, solved = job.get()
			individual.fitness = fitness
			avg_fitness += individual.fitness
			self.env.evaluations += 1

			if solved:
				self.env.solved = True

			if not best_individual or individual.fitness > best_individual.fitness:
				best_individual = individual

		avg_fitness /= len(individuals)
		return best_individual, best_individual.fitness, avg_fitness

	def __del__(self):
		self.pool.close()
		self.pool.join()
