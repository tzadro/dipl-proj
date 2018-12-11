# todo: add interspecies mating probability;

from Config import config
from NEAT import NEAT
import Environments
import Interface
import statistics

env = Environments.XORProblem()
config.update(env.num_inputs, env.num_outputs, env.action_space_discrete, env.action_space_high, env.action_space_low)

algorithm = NEAT(env)
networkVisualizer = Interface.NetworkVisualizer()

num_evaluations = []
best_fitnesses = []
avg_fitnesses = []

for run in range(config.num_runs):
	for i in range(config.num_iter):
		best_individual, best_fitness, avg_fitness = algorithm.epoch()
		best_fitnesses.append(best_fitness)
		avg_fitnesses.append(avg_fitness)

		if config.visualize_best_networks:
			for individual in algorithm.population.individuals:
				networkVisualizer.update_node_positions(individual.connections, individual.nodes)

		if env.solved:
			if config.visualize_best_networks:
				networkVisualizer.visualize_network(best_individual.connections)

			num_evaluations.append(env.evaluations)
			break

	Interface.plot_overall_fitness(best_fitnesses, avg_fitnesses)

	env.evaluations = 0
	env.solved = False
	best_fitnesses = []
	avg_fitnesses = []
	algorithm.reset()

if config.num_runs > 1:
	avg_num_evaluations = sum(num_evaluations) / len(num_evaluations)
	stdev_num_evaluations = statistics.stdev(num_evaluations)
	print('Num iterations:\tavg: {:.2f},\tstdev: {:.2f},\tfrom: {:d} runs'.format(avg_num_evaluations, stdev_num_evaluations, len(num_evaluations)))

env.close()
