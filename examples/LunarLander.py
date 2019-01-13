from core.config import config
from core.environments import LunarLander
from core.statistics import Statistics
from core import neat, interface

config.pop_size = 300
config.c1 = 2.0
config.c2 = 2.0
config.compatibility_threshold = 1.0
config.ct_step = 0.15
config.ct_min_val = 0.4
config.ct_max_val = 4.6
config.new_node_probability = 0.06
config.new_connection_probability = 0.1
config.verbose = True

env = LunarLander()
stats = Statistics()
algorithm = neat.NEAT(env.evaluate)
network_visualizer = interface.NetworkVisualizer()

for i in range(config.num_iter):
	interface.log('Generation: {:d}'.format(i))

	best_individual = algorithm.epoch(stats)
	# env.seed += 1

	if config.visualize_best_networks:
		for spec in algorithm.population.species:
			for individual in spec.individuals:
				network_visualizer.update_node_positions(individual.connections, individual.nodes)

		if i % config.visualize_every == 0:
			network_visualizer.visualize_network(best_individual.connections)

	avg_score = env.evaluate(best_individual, True)
	interface.log('Solve attempt avg_score: {:.2f}'.format(avg_score))

	if avg_score >= 200:
		if config.visualize_best_networks and i % config.visualize_every != 0:
			network_visualizer.visualize_network(best_individual.connections)

		interface.log('Solved after {:d} generations'.format(i))
		break

interface.plot_overall_fitness(stats.best_fitnesses, stats.avg_fitnesses, stats.stdev_fitnesses)
interface.plot_distances(stats.avg_num_hidden_nodesg, stats.stdev_num_hidden_nodes, stats.avg_num_connections, stats.stdev_num_connections)
interface.plot_species_sizes(stats.species_sizes, stats.compatibility_thresholds)
interface.plot_distances(stats.avg_Es, stats.avg_Ds, stats.avg_weight_diffs)

input('Press any key to demonstrate best individual: ')
env.seed = 0
env.evaluate(best_individual)
