import gym
import ple
import random
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
		self.game = ple.games.pixelcopter.Pixelcopter()
		self.env = ple.PLE(self.game, fps=60, display_screen=True, force_fps=True)
		self.env.init()

		self.num_inputs = 7
		self.num_outputs = 2
		self.action_space_discrete = True
		self.action_space_high = None
		self.action_space_low = None

		self.action_set = self.env.getActionSet()

	def reset(self):
		self.env.reset_game()

		observation = self.game.getGameState().values()
		return observation

	def step(self, action_index):
		action = self.action_set[action_index]

		reward = self.env.act(action)
		observation = self.game.getGameState().values()
		done = self.env.game_over()
		info = None
		return observation, reward, done, info

	def close(self):
		return


class TestEnvironment:
	def __init__(self):
		self.possible_observations = [[1., 0., 0., 0.],
									  [0., 1., 0., 0.],
									  [0., 0., 1., 0.],
									  [0., 0., 0., 1.]]
		self.last_observation_index = None

		self.num_inputs = 4
		self.num_outputs = 4
		self.action_space_discrete = False
		self.action_space_high = np.array([0., 0., 0., 0.])
		self.action_space_low = np.array([1., 1., 1., 1.])

	def reset(self):
		self.last_observation_index = random.randrange(4)
		observation = self.possible_observations[self.last_observation_index]
		return observation

	def step(self, output):
		reward = output[self.last_observation_index]
		self.last_observation_index = random.randrange(4)
		observation = self.possible_observations[self.last_observation_index]
		done = np.argmax(output) != self.last_observation_index
		info = None
		return observation, reward, done, info

	def close(self):
		return
