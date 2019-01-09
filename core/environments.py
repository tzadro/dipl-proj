from core.config import config
from core.phenotype import Phenotype
from core import utility
import gym
import ple


class AbstractEnvironment:
	def __init__(self):
		num_inputs, num_outputs = 0, 0
		config.update(num_inputs, num_outputs)

		pass

	def evaluate(self, individual):
		pass

	def reset(self):
		pass


class XORProblem(AbstractEnvironment):
	def __init__(self):
		num_inputs, num_outputs = 2, 1
		config.update(num_inputs, num_outputs)

		self.observations = [[0., 0.],
							 [0., 1.],
							 [1., 0.],
							 [1., 1.]]
		self.solutions = [0., 1., 1., 0.]

		self.evaluations = 0
		self.solved = False

	def evaluate(self, individual):
		self.evaluations += 1

		phenotype = Phenotype(individual.connections.values(), individual.nodes.values())

		correct_solutions = True
		fitness = 4

		for observation, solution in zip(self.observations, self.solutions):
			phenotype.flush()
			output = phenotype.forward(observation)
			result = output[0]

			correct_solutions = correct_solutions and round(result) == solution
			fitness -= (solution - result)**2

		self.solved = self.solved or correct_solutions
		return fitness

	def reset(self):
		self.evaluations = 0
		self.solved = False


class Pixelcopter(AbstractEnvironment):
	def __init__(self):
		num_inputs, num_outputs = 7, 1
		config.update(num_inputs, num_outputs)

		self.game = ple.games.pixelcopter.Pixelcopter(width=144, height=144)
		self.env = ple.PLE(self.game, fps=240, display_screen=False, force_fps=True)
		self.action_set = self.env.getActionSet()
		self.env.init()

	def evaluate(self, individual):
		phenotype = Phenotype(individual.connections.values(), individual.nodes.values())

		fitness = 0

		self.env.reset_game()
		observation = self.game.getGameState().values()
		while True:
			output = phenotype.forward(observation)
			action_index = 1 if output[0] > 0.5 else 0
			action = self.action_set[action_index]

			reward = self.env.act(action)
			observation = self.game.getGameState().values()
			done = self.env.game_over()

			fitness += reward

			if done:
				break

		return fitness


class LunarLander(AbstractEnvironment):
	def __init__(self):
		num_inputs, num_outputs = 8, 4
		config.update(num_inputs, num_outputs)

		self.env = gym.make('LunarLander-v2')
		self.seed = 0

	def evaluate(self, individual, num_times=1, fixed_seed=True):
		phenotype = Phenotype(individual.connections.values(), individual.nodes.values())

		fitnesses = []

		for _ in range(num_times):
			phenotype.flush()
			
			fitness = 0

			if fixed_seed:
				# ensures every run starts with same observation
				self.env.seed(self.seed)

			observation = self.env.reset()
			while True:
				self.env.render()

				output = phenotype.forward(observation)
				action = output.index(max(output))
				observation, reward, done, info = self.env.step(action)

				fitness += reward

				if done:
					break

			fitnesses.append(fitness)

		return sum(fitnesses) / num_times


class HalfCheetah(AbstractEnvironment):
	def __init__(self):
		num_inputs, num_outputs = 17, 6
		config.update(num_inputs, num_outputs)

		self.env = gym.make('HalfCheetah-v2')
		self.seed = 0

	def evaluate(self, individual, fixed_seed=True):
		phenotype = Phenotype(individual.connections.values(), individual.nodes.values())

		fitness = 0

		if fixed_seed:
			# ensures every run starts with same observation
			self.env.seed(self.seed)

		observation = self.env.reset()
		while True:
			self.env.render()

			output = phenotype.forward(observation)
			action = utility.scale(output, self.env.action_space.low, self.env.action_space.high)
			observation, reward, done, info = self.env.step(action)

			fitness += reward

			if done:
				break

		return fitness

	def __del__(self):
		self.env.close()
