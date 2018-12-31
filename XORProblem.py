from config import config
from environments import XORProblem
import evaluators
import neat
import interface
import statistics
import time

if __name__ == '__main__':
	env = XORProblem()
	config.update(env.num_inputs, env.num_outputs)
	evaluator = evaluators.ParallelEvaluator(env)
	algorithm = neat.NewNEAT(evaluator.evaluate_all)
	network_visualizer = interface.NetworkVisualizer()

	num_evaluations = []
	best_fitnesses = []
	avg_fitnesses = []

	start_time = time.time()

	for run in range(config.num_runs):
		interface.log("Run {:d}".format(run))

		for i in range(config.num_iter):
			interface.log("\tGeneration {:d}".format(i))

			best_individual, best_fitness, avg_fitness = algorithm.epoch()
			best_fitnesses.append(best_fitness)
			avg_fitnesses.append(avg_fitness)

			if config.visualize_best_networks:
				for individual in algorithm.population.individuals:
					network_visualizer.update_node_positions(individual.connections, individual.nodes)

			if env.solved:
				num_evaluations.append(env.evaluations)
				break

		if config.visualize_best_networks:
			network_visualizer.visualize_network(best_individual.connections)

		interface.plot_overall_fitness(best_fitnesses, avg_fitnesses)

		env.reset()

		best_fitnesses = []
		avg_fitnesses = []
		algorithm.reset()

	elapsed_time = time.time() - start_time
	print("elapsed_time", elapsed_time)

	if config.num_runs > 1 and len(num_evaluations) > 1:
		avg_num_evaluations = sum(num_evaluations) / len(num_evaluations)
		stdev_num_evaluations = statistics.stdev(num_evaluations)
		print('Num evaluations:\tavg: {:.2f} (~{:d} generations),\tstdev: {:.2f},\tfrom: {:d} runs'.format(avg_num_evaluations, round(avg_num_evaluations / config.pop_size), stdev_num_evaluations, len(num_evaluations)))
