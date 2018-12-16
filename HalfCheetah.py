from config import config
from environments import HalfCheetah
from neat import NEAT
import interface

env = HalfCheetah()
config.update(env.num_inputs, env.num_outputs)
algorithm = NEAT(env.evaluate)
networkVisualizer = interface.NetworkVisualizer()

best_fitnesses = []
avg_fitnesses = []

for i in range(config.num_iter):
	best_individual, best_fitness, avg_fitness = algorithm.epoch()
	best_fitnesses.append(best_fitness)
	avg_fitnesses.append(avg_fitness)

	if config.visualize_best_networks:
		for individual in algorithm.population.individuals:
			networkVisualizer.update_node_positions(individual.connections, individual.nodes)

		if i % config.visualize_every == 0:
			networkVisualizer.visualize_network(best_individual.connections)

interface.plot_overall_fitness(best_fitnesses, avg_fitnesses)

env.close()
