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
	best_fitness = population.evaluate_fitness(env)
	print('Generation: {:d}, num_individuals: {:d}, best fitness: {:.2f}'.format(i, len(population.individuals), best_fitness))
	population.breed_new_generation()

env.close()
