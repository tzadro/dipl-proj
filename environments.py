from config import config
from phenotype import Phenotype
import utility
import gym
"""
import ple
import math
import numpy as np

# todo: needs refactor
class CartPole:
	def __init__(self):
		env_name = 'CartPole-v0'
		self.env = gym.make(env_name)

		self.solved = False
		self.evaluations = 0
		self.num_inputs = self.env.observation_space.shape[0]
		self.num_outputs = self.env.action_space.n
		self.action_space_discrete = True
		self.action_space_high = None
		self.action_space_low = None

	def reset(self):
		self.evaluations += 1
		return self.env.reset()

	def step(self, action):
		self.env.render()
		return self.env.step(action)

	def close(self):
		self.env.close()


# todo: needs refactor
class Pixelcopter:
	def __init__(self):
		self.game = ple.games.pixelcopter.Pixelcopter(width=144, height=144)
		self.env = ple.PLE(self.game, fps=60, display_screen=True, force_fps=True)
		self.env.init()

		self.solved = False
		self.evaluations = 0
		self.num_inputs = 7
		self.num_outputs = 2
		self.action_space_discrete = True
		self.action_space_high = None
		self.action_space_low = None

		self.action_set = self.env.getActionSet()

		# todo: remove after implemented normalized inputs
		self.avg_observations = np.array([0., 0., 0., 0., 0., 0., 0.])
		self.min_observations = np.array([math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf])
		self.max_observations = np.array([-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf])
		self.num_runs = 0

	def reset(self):
		self.env.reset_game()
		self.evaluations += 1

		observation = self.game.getGameState().values()
		return observation

	def step(self, action_index, _):
		action = self.action_set[action_index]

		reward = self.env.act(action)
		observation = self.game.getGameState().values()
		for i, value in enumerate(observation):
			self.avg_observations[i] += value
			self.min_observations[i] = min(value, self.min_observations[i])
			self.max_observations[i] = max(value, self.max_observations[i])
		self.num_runs += 1
		done = self.env.game_over()
		info = None
		return observation, reward, done, info

	def close(self):
		self.avg_observations /= self.num_runs
		for i, key in enumerate(list(self.game.getGameState().keys())):
			print('avg_value: {:.2f}, \tmin_value: {:.2f}, \tmax_value: {:.2f}, \tkey: '.format(self.avg_observations[i], self.min_observations[i], self.max_observations[i]) + key)
"""


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


class HalfCheetah(AbstractEnvironment):
	def __init__(self):
		num_inputs, num_outputs = 17, 6
		config.update(num_inputs, num_outputs)

		self.env = gym.make('HalfCheetah-v2')

	def evaluate(self, individual):
		phenotype = Phenotype(individual.connections.values(), individual.nodes.values())

		fitness = 0

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
