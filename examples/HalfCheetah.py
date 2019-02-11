from core.config import config
from core.environments import HalfCheetah
from core.statistics import Statistics
from core import neat, interface, utility

config.pop_size = 300
config.c1 = 2.0
config.c2 = 2.0
config.compatibility_threshold = 1.0
config.ct_step = 0.1
config.ct_min_val = 0.1
config.ct_max_val = 2.1
config.new_node_probability = 0.06
config.new_connection_probability = 0.1
config.verbose = True

env = HalfCheetah()
stats = Statistics()
algorithm = neat.NEAT(env.evaluate)
network_visualizer = interface.NetworkVisualizer()

agent_file_name = 'half-cheetah-agent.pickle'
video_file_name = None  # 'half-cheetah-demo.mp4' causes a bug
demonstrate = interface.demonstrate_if_exists(agent_file_name, env, video_file_name)

if demonstrate:
	exit()

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

interface.print_info(best_individual)
interface.plot_overall_fitness(stats.best_fitnesses, stats.avg_fitnesses, stats.stdev_fitnesses)
interface.plot_structures(stats.avg_num_hidden_nodes, stats.stdev_num_hidden_nodes, stats.avg_num_connections, stats.stdev_num_connections)
interface.plot_species_sizes(stats.species_sizes, stats.compatibility_thresholds)
interface.plot_distances(stats.avg_Es, stats.avg_Ds, stats.avg_weight_diffs)

utility.save(agent_file_name, best_individual)
