# todo: add interspecies mating probability;

from Config import config
from NEAT import NEAT
import Environments
import Interface
import statistics

env = Environments.XORProblem()
config.update(env.num_inputs, env.num_outputs, env.action_space_discrete, env.action_space_high, env.action_space_low)

algorithm = NEAT(env)

num_evaluations = []
for run in range(config.num_runs):
	num_evaluations.append(algorithm.run())
	Interface.plot_overall_fitness(algorithm.best_fitnesses, algorithm.avg_fitnesses)
	algorithm.reset()

if config.num_runs > 1:
	avg_num_evaluations = sum(num_evaluations) / len(num_evaluations)
	stdev_num_evaluations = statistics.stdev(num_evaluations)
	print('Num iterations:\tavg: {:.2f},\tstdev: {:.2f}'.format(avg_num_evaluations, stdev_num_evaluations))

env.close()
