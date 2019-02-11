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
		# track number of evaluations made
		self.evaluations += 1

		# create phenotype from individual's connection and node genes
		phenotype = Phenotype(individual.connections.values(), individual.nodes.values())

		correct_solutions = True
		fitness = 4

		# for every one of four different cases feed the network input and calculate error based on the output
		for observation, solution in zip(self.observations, self.solutions):
			phenotype.flush()
			output = phenotype.forward(observation)
			result = output[0]

			correct_solutions = correct_solutions and round(result) == solution
			fitness -= (solution - result)**2

		# if solution made no errors the environment is considered solved
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

	def evaluate(self, individual, solve_attempt=False, render=False, video_file_name=None):
		if video_file_name:
			video_recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(self.env, path=video_file_name)

		phenotype = Phenotype(individual.connections.values(), individual.nodes.values())

		fitnesses = []

		num_times = 100 if solve_attempt else 1
		for i in range(num_times):
			phenotype.flush()
			
			fitness = 0

			if solve_attempt:
				# if solve attempt test first 100 seeds
				self.env.seed(i)
			else:
				# else run fixed seed
				self.env.seed(self.seed)

			observation = self.env.reset()
			while True:
				if render:
					self.env.render()

				if video_file_name:
					video_recorder.capture_frame()

				output = phenotype.forward(observation)
				action = output.index(max(output))
				observation, reward, done, info = self.env.step(action)

				fitness += reward

				if done:
					break

			fitnesses.append(fitness)

		if video_file_name:
			video_recorder.close()

		return sum(fitnesses) / num_times

	def __del__(self):
		self.env.close()


class HalfCheetah(AbstractEnvironment):
	def __init__(self):
		num_inputs, num_outputs = 17, 6
		config.update(num_inputs, num_outputs)

		self.env = gym.make('HalfCheetah-v2')
		self.seed = 0

	def evaluate(self, individual, fixed_seed=True, render=False, video_file_name=None):
		if video_file_name:
			video_recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(self.env, path=video_file_name)

		# create phenotype from individual's connection and node genes
		phenotype = Phenotype(individual.connections.values(), individual.nodes.values())

		fitness = 0

		if fixed_seed:
			# ensures every run starts with same observation
			self.env.seed(self.seed)

		# start new game
		observation = self.env.reset()
		while True:
			if render:
				# displays game state on screen
				self.env.render()

			if video_file_name:
				# save game state as a video
				video_recorder.capture_frame()

			# feed forward neural network with observation
			output = phenotype.forward(observation)
			# scale output to get action
			action = utility.scale(output, self.env.action_space.low, self.env.action_space.high)
			# take step with given action
			observation, reward, done, info = self.env.step(action)

			# accumulate rewards to get individual's fitness
			fitness += reward

			# stop if game is over
			if done:
				break

		if video_file_name:
			video_recorder.close()

		return fitness

	def __del__(self):
		self.env.close()
