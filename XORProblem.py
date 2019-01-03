from config import config
from environments import XORProblem
from statistics import Statistics
import neat
import interface

env = XORProblem()
stats = Statistics()
algorithm = neat.NewNEAT(env.evaluate, stats)
network_visualizer = interface.NetworkVisualizer()

for run in range(config.num_runs):
	interface.log("Run {:d}".format(run))

	for i in range(config.num_iter):
		interface.log("\tGeneration {:d}".format(i))

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
	interface.plot_species_sizes(stats.species_sizes)

	env.reset()
	algorithm.reset()

if config.num_runs > 1 and len(stats.num_evaluations) > 1:
	interface.print_evaluation_stats(stats.num_evaluations, stats.num_hidden_nodes, stats.num_connections)
