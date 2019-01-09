from core.config import config
from core.environments import HalfCheetah
from core.statistics import Statistics
from core import neat, interface

env = HalfCheetah()
stats = Statistics()
algorithm = neat.NEAT(env.evaluate, stats)
network_visualizer = interface.NetworkVisualizer()

for i in range(config.num_iter):
	best_individual = algorithm.epoch()
	env.seed += 1

	print('Generation: {:d}, best_score: {:.2f}, avg_score: {:.2f}'.format(i, stats.best_fitnesses[-1], stats.avg_fitnesses[-1]))

	if config.visualize_best_networks:
		for individual in algorithm.population.individuals:
			network_visualizer.update_node_positions(individual.connections, individual.nodes)

		if i % config.visualize_every == 0:
			network_visualizer.visualize_network(best_individual.connections)

interface.plot_overall_fitness(stats.best_fitnesses, stats.avg_fitnesses, stats.stdev_fitnesses)
interface.plot_species_sizes(stats.species_sizes, stats.compatibility_thresholds)

input('Press any key to demonstrate best individual: ')
env.evaluate(best_individual)
