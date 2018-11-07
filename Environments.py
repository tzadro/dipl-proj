import gym
import ple
import math
import numpy as np


class CartPole:
	def __init__(self):
		env_name = 'CartPole-v0'
		self.env = gym.make(env_name)

		self.num_inputs = self.env.observation_space.shape[0]
		self.num_outputs = self.env.action_space.n
		self.action_space_discrete = True
		self.action_space_high = None
		self.action_space_low = None

	def reset(self):
		return self.env.reset()

	def step(self, action):
		self.env.render()
		return self.env.step(action)

	def close(self):
		self.env.close()


class Pixelcopter:
	def __init__(self):
		self.game = ple.games.pixelcopter.Pixelcopter(width=144, height=144)
		self.env = ple.PLE(self.game, fps=60, display_screen=True, force_fps=True)
		self.env.init()

		self.solved = False
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
		return


class XORProblem:
	def __init__(self):
		self.observations = [[0., 0., 1.],
							 [0., 1., 1.],
							 [1., 0., 1.],
							 [1., 1., 1.]]
		self.solutions = [0., 1., 1., 0.]

		self.outputs = None
		self.progress = None
		self.error_sum = None

		self.solved = False
		self.num_inputs = 3
		self.num_outputs = 1
		self.action_space_discrete = False
		self.action_space_high = np.array([1.])
		self.action_space_low = np.array([0.])

	def reset(self):
		self.outputs = []
		self.progress = 0
		self.error_sum = 0
		return self.observations[self.progress]

	def step(self, output, _):
		self.outputs.append(output[0])
		self.error_sum += abs(self.solutions[self.progress] - output[0])

		if self.progress == 3:
			reward = (4 - self.error_sum)**2
			observation = None
			done = True

			correct = [e > 0.5 for e in self.outputs] == self.solutions
			self.solved = self.solved or correct
		else:
			self.progress += 1
			reward = 0
			observation = self.observations[self.progress]
			done = False

		info = None
		return observation, reward, done, info

	def close(self):
		return


class TestEnvironment:
	def __init__(self):
		self.observation = [1., 1., 1., 1.]

		self.solved = False
		self.num_inputs = 4
		self.num_outputs = 4
		self.action_space_discrete = False
		self.action_space_high = np.array([0., 0., 0., 0.])
		self.action_space_low = np.array([1., 1., 1., 1.])

	def reset(self):
		return self.observation

	def step(self, _, weights):
		# reward = len(weights)  # fitness = number of connections
		# reward = max(0.001, float(np.sum(weights)))  # fitness = sum of all weights
		reward = max(0.001, float(np.sum(weights))) / len(weights)  # fitness = avg weight
		observation = self.observation
		done = True
		info = None
		return observation, reward, done, info

	def close(self):
		return
