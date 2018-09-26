# todo: add interspecies mating probability;
# todo: remove species after their best fitness doesn't improve for some number of turns

from Config import config
from Population import Population
import gym

env_name = 'CartPole-v0'
env = gym.make(env_name)
config.update(env.observation_space, env.action_space)

population = Population()
for i in range(config.num_iter):
	population.speciate()
	best_fitness, avg_fitness = population.evaluate_fitness(env)
	population.adjust_fitness()
	population.assign_num_children()

	print('Generation: {:d}, num_individuals: {:d}, best_score: {:.2f}, avg_score: {:.2f}'.format(i, len(population.individuals), best_fitness, avg_fitness))
	print('Num organisms with more than default number of connections: {:d}'.format(sum([len(individual.connections.values()) > config.starting_num_connections for individual in population.individuals])))
	for j, spec in enumerate(population.species):
		print('\tSpecies: {:d}'.format(j))
		print('\t\tfitness: {:.2f}'.format(spec.fitness))
		print('\t\tnum_individuals: {:d}, num_children: {:d}'.format(len(spec.individuals), spec.num_children))
		best_adjusted_fitness = spec.individuals[0].adjusted_fitness
		avg_adjusted_fitness = sum([individual.adjusted_fitness for individual in spec.individuals]) / len(spec.individuals)
		print('\t\tbest_adjusted_fitness: {:.2f}, avg_adjusted_fitness: {:.2f}'.format(best_adjusted_fitness, avg_adjusted_fitness))

	population.remove_worst()
	population.breed_new_generation()

env.close()
