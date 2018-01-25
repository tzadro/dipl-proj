# todo: add interspecies mating probability; remove species after their best fitness doesn't improve for some number of turns

import gym
import math
import numpy as np
from copy import deepcopy
from random import random, randrange, gauss


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def max_num_connections(n):
    return n * (n - 1) / 2


def check_if_path(from_node, to_node, connections, checked):
    for connection in connections.values():
        if not connection.enabled:
            continue

        if connection.from_key == from_node and connection.to_key == to_node:
            return True

    checked[from_node] = True

    for connection in connections.values():
        if not connection.enabled:
            continue

        if connection.from_key == from_node and connection.to_key not in checked and check_if_path(connection.to_key,
                                                                                                   to_node, connections,
                                                                                                   checked):
            return True

    return False


def check_if_path2(from_key, to_key, neurons, checked):
    if to_key in neurons[from_key].outgoing_keys:
        return True

    checked[from_key] = True

    for key in neurons[from_key].outgoing_keys:
        if key not in checked and check_if_path2(key, to_key, neurons, checked):
            return True

    return False


# todo: are disabled genes included?
def distance(individual1, individual2):
    connections1 = individual1.connections
    connections2 = individual2.connections

    innovation_numbers = innovation_numbers_union(connections1, connections2)
    max_common = min(individual1.max_innovation, individual1.max_innovation)

    weight_diffs = []
    E = 0
    D = 0
    N = max(len(connections1), len(connections2))

    for innovation_number in innovation_numbers:
        if innovation_number in connections1 and innovation_number in connections2:
            weight_diff = abs(connections1[innovation_number].weight - connections2[innovation_number].weight)
            weight_diffs.append(weight_diff)
        else:
            if innovation_number > max_common:
                E = E + 1
            else:
                D = D + 1

    delta = (config.c1 * E) / N + (config.c2 * D) / N + config.c3 * sum(weight_diffs) / len(weight_diffs)
    return delta


def innovation_numbers_union(connections1, connections2):
    innovation_numbers1 = [connection.innovation_number for connection in connections1.values()]
    innovation_numbers2 = [connection.innovation_number for connection in connections2.values()]

    innovation_numbers = set().union(innovation_numbers1).union(innovation_numbers2)
    return innovation_numbers


def crossover(parents):
    parent1 = parents[0]
    parent2 = parents[1]

    parent1_connections = parent1.connections
    parent2_connections = parent2.connections

    innovation_numbers = innovation_numbers_union(parent1_connections, parent2_connections)

    child_connections = {}
    child_nodes = {}
    node_pairs = []

    for innovation_number in innovation_numbers:
        if innovation_number in parent1_connections and innovation_number in parent2_connections:
            connection1 = parent1_connections[innovation_number]
            connection2 = parent2_connections[innovation_number]

            if random() < 0.5:
                new_connection = deepcopy(connection1)
            else:
                new_connection = deepcopy(connection2)

            if not connection1.enabled or not connection2.enabled:
                new_connection.enabled = random() < (1 - config.disable_probability)

            child_connections[innovation_number] = new_connection

            if new_connection.from_key not in child_nodes:
                child_nodes[new_connection.from_key] = Node(new_connection.from_key)

            if new_connection.to_key not in child_nodes:
                child_nodes[new_connection.to_key] = Node(new_connection.to_key)

            node_pairs.append((new_connection.from_key, new_connection.to_key))
        elif innovation_number in parent1_connections and not innovation_number in parent2_connections:
            connection1 = parent1_connections[innovation_number]
            new_connection = deepcopy(connection1)

            if not connection1.enabled:
                new_connection.enabled = random() < (1 - config.disable_probability)

            child_connections[innovation_number] = new_connection

            if new_connection.from_key not in child_nodes:
                child_nodes[new_connection.from_key] = Node(new_connection.from_key)

            if new_connection.to_key not in child_nodes:
                child_nodes[new_connection.to_key] = Node(new_connection.to_key)

            node_pairs.append((new_connection.from_key, new_connection.to_key))
        elif not innovation_number in parent1_connections and innovation_number in parent2_connections:
            connection2 = parent2_connections[innovation_number]
            new_connection = deepcopy(connection2)

            if not connection2.enabled:
                new_connection.enabled = random() < (1 - config.disable_probability)

            child_connections[innovation_number] = new_connection

            if new_connection.from_key not in child_nodes:
                child_nodes[new_connection.from_key] = Node(new_connection.from_key)

            if new_connection.to_key not in child_nodes:
                child_nodes[new_connection.to_key] = Node(new_connection.to_key)

            node_pairs.append((new_connection.from_key, new_connection.to_key))

    max_innovation = max(parent1.max_innovation, parent2.max_innovation)
    max_node = max(parent1.max_node, parent2.max_node)
    child = Individual(child_connections, child_nodes, node_pairs, max_innovation, max_node)
    return child


class Connection:  # Gene
    def __init__(self, innovation_number, from_key, to_key, weight, enabled):
        self.innovation_number = innovation_number
        self.from_key = from_key
        self.to_key = to_key
        self.weight = weight
        self.enabled = enabled


class Node:
    def __init__(self, key):
        self.key = key
        self.value = 0


class Individual:  # Genome
    def __init__(self, connections=None, nodes=None, node_pairs=None, max_innovation=None,
                 max_node=None):
        self.nodes = {}
        self.connections = {}
        self.fitness = 0
        self.max_innovation = 0
        self.max_node = 0
        self.node_pairs = []

        if connections is None or nodes is None or node_pairs is None or max_innovation is None or max_node is None:
            self.configure_new()
        else:
            self.connections = connections
            self.nodes = nodes
            self.node_pairs = node_pairs
            self.max_innovation = max_innovation
            self.max_node = max_node

    def configure_new(self):
        for key in config.input_keys:
            self.nodes[key] = Node(key)
            self.max_node = self.max_node + 1

        for key in config.output_keys:
            self.nodes[key] = Node(key)
            self.max_node = self.max_node + 1

        for input_key in config.input_keys:
            for output_key in config.output_keys:
                new_connection = Connection(self.max_innovation, input_key, output_key,
                                            gauss(config.new_mu, config.new_sigma), True)
                self.connections[self.max_innovation] = new_connection
                self.max_innovation = self.max_innovation + 1
                self.node_pairs.append((input_key, output_key))

    def evaluate_fitness(self):
        phenotype = Phenotype(self.connections)

        observation = env.reset()

        while True:
            env.render()

            output_key = phenotype.forward(observation)
            observation, reward, done, info = env.step(config.actions[output_key])
            self.fitness = self.fitness + reward

            if done:
                return self.fitness

    def mutate(self, generation_innovations):
        if random() < config.connection_mutation_probability:
            self.mutate_connections()

        if random() < config.new_connection_probability:
            self.new_connection(generation_innovations)

        if random() < config.new_node_probability:
            self.new_node()

    def mutate_connections(self):
        for connection in self.connections.values():
            if random() < config.perturbation_probability:
                connection.weight = connection.weight + gauss(config.step_mu,
                                                              config.step_sigma)  # random() * 2 * config.step - config.step
            else:
                connection.weight = random() * 2 - 1

    def new_connection(self, generation_innovations):
        num_connections = len(self.node_pairs)
        num_nodes = len(list(self.nodes.keys()))
        num_inputs = len(config.input_keys)
        num_outputs = len(config.output_keys)

        node_list = list(self.nodes.values())

        if num_connections == max_num_connections(num_nodes) - (
                max_num_connections(num_inputs) + max_num_connections(num_outputs)):
            return

        while True:
            node1 = node_list[randrange(num_inputs + num_outputs, num_nodes)]
            node2 = node_list[randrange(num_nodes)]

            if node1.key == node2.key:
                continue

            pair = (min(node1.key, node2.key), max(node1.key, node2.key))

            if pair in self.node_pairs:
                continue

            self.node_pairs.append(pair)

            if node2.key in config.input_keys or check_if_path(node2.key, node1.key, self.connections, {}) or (
            node2.key, node1.key) in generation_innovations:
                temp = node1
                node1 = node2
                node2 = temp

            if (node1.key, node2.key) in generation_innovations:
                innovation_number = generation_innovations[(node1.key, node2.key)]
                self.max_innovation = max(self.max_innovation, innovation_number)
            else:
                innovation_number = config.innovation_number
                generation_innovations[(node1.key, node2.key)] = innovation_number
                self.max_innovation = config.innovation_number
                config.innovation_number = config.innovation_number + 1

            new_connection = Connection(innovation_number, node1.key, node2.key, random() * 2 - 1, True)
            self.connections[innovation_number] = new_connection
            return

    def new_node(self):
        connections_values = list(self.connections.values())

        connection = connections_values[randrange(len(connections_values))]
        connection.enabled = False

        new_node = Node(config.node_key)
        self.max_node = config.node_key
        config.node_key = config.node_key + 1
        self.nodes[new_node.key] = new_node

        new_connection1 = Connection(config.innovation_number, connection.from_key, new_node.key, 1.0, True)
        self.max_innovation = config.innovation_number
        self.connections[config.innovation_number] = new_connection1
        config.innovation_number = config.innovation_number + 1

        new_connection2 = Connection(config.innovation_number, new_node.key, connection.to_key, connection.weight, True)
        self.max_innovation = config.innovation_number
        self.connections[config.innovation_number] = new_connection2
        config.innovation_number = config.innovation_number + 1


class Phenotype:  # Neural network
    def __init__(self, connections):
        self.neurons = {}

        for connection in connections.values():
            if not connection.enabled:
                continue

            if connection.from_key not in self.neurons:
                self.neurons[connection.from_key] = Neuron(connection.from_key)

            from_neuron = self.neurons[connection.from_key]
            from_neuron.outgoing_keys.append(connection.to_key)

            if connection.to_key not in self.neurons:
                self.neurons[connection.to_key] = Neuron(connection.to_key)

            to_neuron = self.neurons[connection.to_key]
            to_neuron.incoming_connections.append(connection)
            if not check_if_path2(connection.to_key, connection.from_key, self.neurons, {}):
                to_neuron.num_accepts_before_firing = to_neuron.num_accepts_before_firing + 1

    def forward(self, inputs):
        for neuron in self.neurons.values():
            neuron.reset()

        for key, value in zip(config.input_keys, inputs):
            self.neurons[key].set_value(value, self.neurons)

        # todo: ugly
        max_key = None
        max_value = -1
        for key in config.output_keys:
            global asd
            asd = self.neurons

            if self.neurons[key].value > max_value:
                max_key = key
                max_value = self.neurons[key].value

        return max_key


class Neuron:
    def __init__(self, key):
        self.key = key
        self.incoming_connections = []
        self.outgoing_keys = []
        self.num_accepts_before_firing = 0
        self.count = None
        self.value = 0

    def reset(self):
        self.count = self.num_accepts_before_firing

    def accept(self, neurons):
        self.count = self.count - 1

        if self.count == 0:
            score = 0

            for connection in self.incoming_connections:
                from_neuron = neurons[connection.from_key]
                score = score + connection.weight * from_neuron.value

            self.value = sigmoid(score)

            self.trigger_outgoing(neurons)

    def set_value(self, value, neurons):
        self.value = value
        self.trigger_outgoing(neurons)

    def trigger_outgoing(self, neurons):
        for key in self.outgoing_keys:
            outgoing_neuron = neurons[key]
            outgoing_neuron.accept(neurons)


class Population:
    def __init__(self):
        self.individuals = [Individual() for _ in range(config.pop_size)]
        self.species = []
        self.max_fitness = -math.inf

    def evaluate_fitness(self):
        for individual in self.individuals:
            self.max_fitness = max(self.max_fitness, individual.evaluate_fitness())

        return self.max_fitness

    def speciate(self):
        sum_species_fitness = 0

        for individual in self.individuals:
            placed = False

            for spec in self.species:
                distance_from_representative = distance(individual, spec.representative)

                if distance_from_representative <= config.compatibility_threshold:
                    spec.add(individual)
                    placed = True
                    break

            if not placed:
                self.species.append(Species(individual))

        for spec in self.species:
            spec.adjust_fitness()
            sum_species_fitness = sum_species_fitness + spec.species_fitness

        for spec in self.species:
            spec.num_children = math.floor(spec.species_fitness / sum_species_fitness * config.pop_size)
            # todo: when should this be called?
            spec.set_representative()

    def breed_new_generation(self):
        children = []
        generation_innovations = {}

        for spec in self.species:
            if spec.num_children == 0:
                continue

            spec.remove_worst()
            if len(spec.individuals) == 0:
                self.species.remove(spec)
                continue

            children = children + [spec.breed_child(generation_innovations) for _ in range(spec.num_children)]
            spec.clear()

        self.individuals = children
        self.max_fitness = 0


class Species:
    def __init__(self, representative):
        self.representative = deepcopy(representative)
        self.current_closest = (representative, 0)
        self.individuals = [representative]
        self.species_fitness = 0
        self.num_children = None

    def add(self, individual):
        self.individuals.append(individual)

    def set_representative(self):
        for individual in self.individuals:
            distance_from_representative = distance(individual, self.representative)
            if distance_from_representative < self.current_closest[1]:
                self.current_closest = (individual, distance_from_representative)

        self.representative = self.current_closest[0]
        self.current_closest = (None, math.inf)

    def adjust_fitness(self):
        for individual in self.individuals:
            individual.fitness = individual.fitness / len(self.individuals)
            self.species_fitness = self.species_fitness + individual.fitness

    def breed_child(self, generation_innovations):
        if random() < config.crossover_probability and len(self.individuals) > 1:
            child = crossover(self.select(2))
        else:
            child = deepcopy(self.select())

        child.mutate(generation_innovations)
        return child

    def select(self, size=None, replace=False):
        fitness_sum = sum([individual.fitness for individual in self.individuals])
        p = [individual.fitness / fitness_sum for individual in self.individuals]
        return np.random.choice(self.individuals, size, replace, p)

    def remove_worst(self):
        def key(element):
            return element.fitness

        if config.num_to_remove >= len(self.individuals):
            self.individuals = []
            return

        self.individuals.sort(key=key)
        for individual in self.individuals[0:config.num_to_remove]:
            self.species_fitness = self.species_fitness - individual.fitness
        self.individuals = self.individuals[config.num_to_remove:]

    def clear(self):
        self.individuals = []
        self.num_children = None
        self.species_fitness = 0


class Config:
    def __init__(self):
        self.connection_mutation_probability = 0.8
        self.perturbation_probability = 0.9
        self.new_node_probability = 0.04
        self.new_connection_probability = 0.08
        self.step = 0.25
        self.innovation_number = 8
        self.node_key = 6
        self.c1 = 1.0
        self.c2 = 1.0
        self.c3 = 0.8
        self.compatibility_threshold = 1.0
        self.crossover_probability = 0.75
        self.disable_probability = 0.75
        self.input_keys = [0, 1, 2, 3]
        self.output_keys = [4, 5]
        self.actions = {
            self.output_keys[0]: 0,
            self.output_keys[1]: 1
        }
        self.pop_size = 30
        self.num_iter = 70
        self.num_to_remove = 2
        self.new_mu = 0
        self.new_sigma = 1
        self.step_mu = 0
        self.step_sigma = 1


config = Config()
env = gym.make('CartPole-v0')

population = Population()
for i in range(config.num_iter):
    print('Generation: {:d}, best fitness: {:.2f}'.format(i, population.evaluate_fitness()))
    population.speciate()
    population.breed_new_generation()

env.render(close=True)
