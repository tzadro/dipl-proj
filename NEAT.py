# todo: add interspecies mating probability;

from Config import config
from Population import Population
import Environments
import Interface

env = Environments.XORProblem()
config.update(env.num_inputs, env.num_outputs, env.action_space_discrete, env.action_space_high, env.action_space_low)

best_fitnesses = []
avg_fitnesses = []

population = Population()
for i in range(config.num_iter):
	population.speciate()

	visualize = i % config.visualize_every == 0
	best_fitness, avg_fitness = population.evaluate_fitness(env, visualize)
	best_fitnesses.append(best_fitness)
	avg_fitnesses.append(avg_fitness)

	population.adjust_fitness()
	population.assign_num_children()

	if config.verbose:
		Interface.verbose(i, population, best_fitness, avg_fitness)

	if env.solved:
		break

	population.remove_worst()
	population.breed_new_generation()

Interface.plot_overall_fitness(best_fitnesses, avg_fitnesses)

env.close()
