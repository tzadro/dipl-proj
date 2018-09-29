import gym
import ple


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
