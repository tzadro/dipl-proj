from core.config import config
from core.environments import XORProblem
from core.statistics import Statistics
from core import neat, interface

env = XORProblem()
stats = Statistics()
algorithm = neat.NEAT(env.evaluate, stats)
network_visualizer = interface.NetworkVisualizer()

for run in range(config.num_runs):
	for i in range(config.num_iter):
		best_individual = algorithm.epoch()

		if config.visualize_best_networks:
			for individual in algorithm.population.individuals:
				network_visualizer.update_node_positions(individual.connections, individual.nodes)

		if env.solved:
			stats.update_run(env.evaluations, best_individual)
			break

	if config.visualize_best_networks:
		network_visualizer.visualize_network(best_individual.connections)

	interface.plot_overall_fitness(stats.best_fitnesses, stats.avg_fitnesses, stats.stdev_fitnesses)
	interface.plot_structures(stats.avg_num_hidden_nodes, stats.stdev_num_hidden_nodes, stats.avg_num_connections, stats.stdev_num_connections)
	interface.plot_species_sizes(stats.species_sizes, stats.compatibility_thresholds)
	interface.plot_distances(stats.avg_Es, stats.avg_Ds, stats.avg_weight_diffs)

	env.reset()
	algorithm.reset()

if config.num_runs > 1 and len(stats.num_evaluations) > 1:
	interface.print_evaluation_stats(stats.num_evaluations, stats.num_hidden_nodes, stats.num_connections)
