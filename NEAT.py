# todo: add interspecies mating probability;
# todo: remove species after their best fitness doesn't improve for some number of turns

from Config import config
from Population import Population
import gym

env = gym.make('CartPole-v0')

population = Population()

for i in range(config.num_iter):
	best_fitness = population.evaluate_fitness(env)
	population.speciate()
	population.breed_new_generation()

	print('Generation: {:d}, best fitness: {:.2f}'.format(i, best_fitness))

env.render(close=True)
