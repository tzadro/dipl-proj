# todo: add interspecies mating probability;
# todo: remove species after their best fitness doesn't improve for some number of turns

from Config import config
from Population import Population
import Environments
import matplotlib.pyplot as plt

env = Environments.Pixelcopter()
config.update(env.num_inputs, env.num_outputs, env.action_space_discrete, env.action_space_high, env.action_space_low)

best_fitnesses = []
avg_fitnesses = []
generation_range = range(config.num_iter)

population = Population()
for i in generation_range:
	population.speciate()

	best_fitness, avg_fitness = population.evaluate_fitness(env)
	best_fitnesses.append(best_fitness)
	avg_fitnesses.append(avg_fitness)

	population.adjust_fitness()
	population.assign_num_children()

	print('Generation: {:d}, num_individuals: {:d}, best_score: {:.2f}, avg_score: {:.2f}'.format(i, len(population.individuals), best_fitness, avg_fitness))
	print('Num organisms with more than default number of connections: {:d}'.format(sum([len(individual.connections.values()) > config.starting_num_connections for individual in population.individuals])))
	for j, spec in enumerate(population.species):
		print('\tSpecies: {:d}'.format(j))
		print('\t\tfitness: {:.2f}'.format(spec.fitness))
		print('\t\tnum_individuals: {:d}, num_children: {:d}'.format(len(spec.individuals), spec.num_children))
		best_adjusted_fitness = spec.individuals[0].adjusted_fitness
		avg_adjusted_fitness = sum([individual.adjusted_fitness for individual in spec.individuals]) / len(spec.individuals)
		print('\t\tbest_adjusted_fitness: {:.2f}, avg_adjusted_fitness: {:.2f}'.format(best_adjusted_fitness, avg_adjusted_fitness))

	population.remove_worst()
	population.breed_new_generation()

plt.plot(generation_range, best_fitnesses, color='red', label='Best score')
plt.plot(generation_range, avg_fitnesses, color='blue', label='Average score')
plt.title('Fitness over generations')
plt.xlabel('Generation')
plt.ylabel('Score')
plt.legend()
plt.show()

env.close()
