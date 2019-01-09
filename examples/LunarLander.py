from core.config import config
from core.environments import LunarLander
from core.statistics import Statistics
from core import neat, interface

env = LunarLander()
stats = Statistics()
algorithm = neat.NEAT(env.evaluate, stats)
networkVisualizer = interface.NetworkVisualizer()

for i in range(config.num_iter):
	best_individual = algorithm.epoch()
	env.seed += 1

	print('Generation: {:d}, best_score: {:.2f}, avg_score: {:.2f}'.format(i, stats.best_fitnesses[-1], stats.avg_fitnesses[-1]))

	if config.visualize_best_networks:
		for individual in algorithm.population.individuals:
			networkVisualizer.update_node_positions(individual.connections, individual.nodes)

		if i % config.visualize_every == 0:
			networkVisualizer.visualize_network(best_individual.connections)

	avg_score = env.evaluate(best_individual, num_times=100, fixed_seed=False)
	print('\tSolve attempt avg_score: {:.2f}'.format(avg_score))

	if avg_score >= 200:
		if config.visualize_best_networks and i % config.visualize_every != 0:
			networkVisualizer.visualize_network(best_individual.connections)

		print('Solved after', i, 'generations')
		break

interface.plot_overall_fitness(stats.best_fitnesses, stats.avg_fitnesses, stats.stdev_fitnesses)
interface.plot_species_sizes(stats.species_sizes, stats.compatibility_thresholds)

input('Press any key to demonstrate best individual: ')
env.evaluate(best_individual, fixed_seed=False)
