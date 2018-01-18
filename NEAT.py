#todo: add interspecies mating probability; remove species after their best fitness doesn't imporove for some number of turns

import gym
import math
import numpy as np
from copy import deepcopy
from random import random, randrange, choice

def sigmoid(x):
    # math.exp() faster than np.exp() for scalars
    return 1 / (1 + math.exp(-x))

def max_num_connections(n):
    return n * (n - 1) / 2

# todo: are disabled genes included?
def distance(individual1, individual2, c1, c2, c3):
    print("entered distance")
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
            # abs() faster than np.abs() for scalars
            weight_diff = abs(connections1[innovation_number].weight - connections2[innovation_number].weight)
            weight_diffs.append(weight_diff)
        else:
            if innovation_number > max_common:
                E = E + 1
            else:
                D = D + 1
    
    # sum() faster than np.sum()
    distance = (c1 * E) / N + (c2 * D) / N + c3 * sum(weight_diffs) / len(weight_diffs)
    print("left distance")
    return distance

def innovation_numbers_union(connections1, connections2):
    innovation_numbers1 = [connection.innovation_number for connection in connections1.values()]
    innovation_numbers2 = [connection.innovation_number for connection in connections2.values()]
    
    innovation_numbers = set().union(innovation_numbers1).union(innovation_numbers2)
    return innovation_numbers

def crossover(parent1, parent2, disable_probability):
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
                new_connection.enabled = random() < (1 - disable_probability)
            
            child_connections[innovation_number] = new_connection
            
            if new_connection.from_id not in child_nodes:
                child_nodes[new_connection.from_id] = Node(new_connection.from_id)
                
            if new_connection.to_id not in child_nodes:
                child_nodes[new_connection.to_id] = Node(new_connection.to_id)
            
            node_pairs.append((new_connection.from_id, new_connection.to_id))
        elif innovation_number in parent1_connections and not innovation_number in parent2_connections:
            connection1 = parent1_connections[innovation_number]
            new_connection = deepcopy(connection1)
            
            if not connection1.enabled:
                new_connection.enabled = random() < (1 - disable_probability)
                
            child_connections[innovation_number] = new_connection
            
            if new_connection.from_id not in child_nodes:
                child_nodes[new_connection.from_id] = Node(new_connection.from_id)
                
            if new_connection.to_id not in child_nodes:
                child_nodes[new_connection.to_id] = Node(new_connection.to_id)
            
            node_pairs.append((new_connection.from_id, new_connection.to_id))
        elif not innovation_number in parent1_connections and innovation_number in parent2_connections:
            connection2 = parent2_connections[innovation_number]
            new_connection = deepcopy(connection2)
            
            if not connection2.enabled:
                new_connection.enabled = random() < (1 - disable_probability)
                
            child_connections[innovation_number] = new_connection
            
            if new_connection.from_id not in child_nodes:
                child_nodes[new_connection.from_id] = Node(new_connection.from_id)
                
            if new_connection.to_id not in child_nodes:
                child_nodes[new_connection.to_id] = Node(new_connection.to_id)
            
            node_pairs.append((new_connection.from_id, new_connection.to_id))
        else:
            print("This should not trigger..")
    
    max_innovation = max(parent1.max_innovation, parent2.max_innovation)
    max_node = max(parent1.max_node, parent2.max_node)
    child = Individual(parent1.input_ids, parent1.output_ids, child_connections, child_nodes, node_pairs, max_innovation, max_node)
    return child
	
class Connection(): # Gene
    def __init__(self, innovation_number, from_id, to_id, weight, enabled):
        self.innovation_number = innovation_number
        self.from_id = from_id
        self.to_id = to_id
        self.weight = weight
        self.enabled = enabled

class Node():
    def __init__(self, id):
        self.id = id
        self.value = 0

class Individual(): # Genome
    def __init__(self, input_ids, output_ids, connections=None, nodes=None, node_pairs=None, max_innovation=None, max_node=None):
        self.nodes = {}
        self.connections = {}
        self.fitness = 0
        self.max_innovation = 0
        self.max_node = 0
        self.node_pairs = []
        
        self.input_ids = input_ids
        self.output_ids = output_ids
        
        if connections is None or nodes is None or node_pairs is None or max_innovation is None or max_node is None:
            self.configure_new()
        else:
            self.connections = connections
            self.nodes = nodes
            self.node_pairs = node_pairs
            self.max_innovation = max_innovation # np.amax(list(self.connections.keys()))
            self.max_node = max_node
        
    def configure_new(self):
        for id in self.input_ids:
            self.nodes[id] = Node(id)
            self.max_node = self.max_node + 1
            
        for id in self.output_ids:
            self.nodes[id] = Node(id)
            self.max_node = self.max_node + 1
        
        for input_id in self.input_ids:
            for output_id in self.output_ids:
                new_connection = Connection(self.max_innovation, input_id, output_id, random() * 2 - 1, True)
                self.connections[self.max_innovation] = new_connection
                self.max_innovation = self.max_innovation + 1
                self.node_pairs.append((input_id, output_id))
    
    def evaluate_fitness(self, env, actions):
        phenotype = Phenotype(self.connections, self.input_ids, self.output_ids)
        
        observation = env.reset()
        
        while True:
            env.render()

            output_id = phenotype.forward(observation)
            observation, reward, done, info = env.step(actions[output_id])
            self.fitness = self.fitness + reward

            if done:
                # print(self.fitness)
                return self.fitness
            
    def mutate(self, config):
        if random() < config.CONNECTION_MUTATION_PROBABILITY:
            self.mutate_connections(config)
        
        if random() < config.NEW_CONNECTION_PROBABILITY:
            self.new_connection(config)
        
        if random() < config.NEW_NODE_PROBABILIY:
            self.new_node(config)
    
    def mutate_connections(self, config):
        for connection in self.connections.values():
            if random() < config.PERTURBATION_PROBABILITY:
                connection.weight = connection.weight + random() * 2 * config.STEP - config.STEP
            else:
                connection.weight = random() * 2 - 1

    def new_connection(self, config):
        num_connections = len(self.node_pairs)
        num_nodes = len(list(self.nodes.keys()))
        num_inputs = len(self.input_ids)
        num_outputs = len(self.output_ids)
        
        if num_connections == max_num_connections(num_nodes) - (max_num_connections(num_inputs) + max_num_connections(num_outputs)):
            return
        
        while True:
            # todo: possible recurrent networks??
            node1 = self.nodes[randrange(num_inputs + num_outputs, num_nodes)]
            node2 = self.nodes[randrange(num_nodes)]
            
            pair = (min(node1.id, node2.id), max(node1.id, node2.id))
            
            if pair in self.node_pairs:
                continue
                
            self.node_pairs.append(pair)
            
            if node2 in self.input_ids:
                temp = node1
                node1 = node2
                node2 = temp
            
            new_connection = Connection(config.INNOVATION_NUMBER, node1, node2, random() * 2 - 1, True)
            self.max_innovation = config.INNOVATION_NUMBER
            config.INNOVATION_NUMBER = config.INNOVATION_NUMBER + 1
    
    def new_node(self, config):
        connections_values = list(self.connections.values())
        
        connection = connections_values[randrange(len(connections_values))]
        connection.enabled = False
        
        new_node = Node(config.NODE_ID)
        self.max_node = config.NODE_ID
        config.NODE_ID = config.NODE_ID + 1
        self.nodes[new_node.id] = new_node
        
        new_connection1 = Connection(config.INNOVATION_NUMBER, connection.from_id, new_node.id, 1.0, True)
        self.max_innovation = config.INNOVATION_NUMBER
        self.connections[config.INNOVATION_NUMBER] = new_connection1
        config.INNOVATION_NUMBER = config.INNOVATION_NUMBER + 1
        
        new_connection2 = Connection(config.INNOVATION_NUMBER, new_node.id, connection.to_id, connection.weight, True)
        self.max_innovation = config.INNOVATION_NUMBER
        self.connections[config.INNOVATION_NUMBER] = new_connection2
        config.INNOVATION_NUMBER = config.INNOVATION_NUMBER + 1
		
class Phenotype(): # Neural network
    def __init__(self, connections, input_ids, output_ids):
        self.neurons = {}
        self.input_ids = input_ids
        self.output_ids = output_ids
        
        for id in input_ids:
            self.neurons[id] = Neuron()
        
        for connection in connections.values():
            if not connection.enabled:
                continue
            
            if connection.to_id not in self.neurons:
                self.neurons[connection.to_id] = Neuron()
            
            self.neurons[connection.to_id].incoming.append(connection)
    
    def forward(self, inputs):
        for id, value in zip(self.input_ids, inputs):
            self.neurons[id].value = value
            
        # todo: not everything is calculated??
        for id, neuron in self.neurons.items():
            if id in self.input_ids:
                continue
            
            sum = 0
            
            for connection in neuron.incoming:
                from_neuron = self.neurons[connection.from_id]
                sum = sum + connection.weight * from_neuron.value
                
            if sum == 0:
                print("nula")
            
            neuron.value = sigmoid(sum)
        
        # todo: ugly
        max_id = None
        max_value = -1
        for id in output_ids:
            if self.neurons[id].value > max_value:
                max_id = id
                max_value = self.neurons[id].value
        
        return max_id

class Population():
    def __init__(self, pop_size, input_ids, output_ids):
        self.pop_size = pop_size
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.individuals = [Individual(input_ids, output_ids) for i in range(pop_size)]
        self.species = []
        self.max_fitness = -math.inf
    
    def evaluate_fitness(self, env, actions):
        for individual in self.individuals:
            self.max_fitness = max(self.max_fitness, individual.evaluate_fitness(env, actions))
            
        return self.max_fitness
            
    def speciate(self, c1, c2, c3, compatibility_threshold):
        sum_species_fitness = 0
        
        for individual in self.individuals:
            placed = False
            
            for spec in self.species:
                distance_from_representative = distance(individual, spec.representative, c1, c2, c3)
                
                if distance_from_representative <= compatibility_threshold:
                    spec.add(individual, distance_from_representative)
                    placed = True
                    break
                    
            if not placed:
                self.species.append(Species(individual))
            
        for spec in self.species:
            spec.adjust_fitness()
            sum_species_fitness = sum_species_fitness + spec.species_fitness

        for spec in self.species:
            spec.num_children = math.floor(spec.species_fitness / sum_species_fitness * self.pop_size)
            # todo: when should this be called?
            spec.set_representative()
    
    def breed_new_generation(self, crossover_probability, disable_probability, config):
        children = []
        
        for spec in self.species:
            children = children + [spec.breed_child(crossover_probability, disable_probability, config) for i in range(spec.num_children)]
            spec.clear()
        
        self.individuals = children
        self.max_fitness = 0

class Species():
    def __init__(self, representative):
        self.representative = deepcopy(representative)
        self.current_closest = (representative, 0)
        self.individuals = [representative]
        self.species_fitness = 0
        self.num_children = None
        
    def add(self, individual, distance_from_representative):
        if distance_from_representative < self.current_closest[1]:
            self.current_closest = (individual, distance_from_representative)
        
        self.individuals.append(individual)
    
    def set_representative(self):
        self.representative = self.current_closest[0]
        self.current_closest = (None, math.inf)
        
    def adjust_fitness(self):
        for individual in self.individuals:
            individual.fitness = individual.fitness / len(self.individuals)
            self.species_fitness = self.species_fitness + individual.fitness
            
    def breed_child(self, crossover_probability, disable_probability, config):
        if random() < crossover_probability:
            child = crossover(self.select(), self.select(), disable_probability)
        else:
            child = deepcopy(self.select())
            
        child.mutate(config)
        return child
    
    def select(self, n=1):
        if n == 1:
            return self.individuals[randrange(len(self.individuals))]
        else:
            return [self.individuals[randrange(len(self.individuals))] for i in range(len(self.individuals))]
        
    def clear(self):
        self.individuals = []
        self.num_children = None
		
class Neuron():
    def __init__(self):
        self.incoming = []
        self.value = 0

class Config():
    def __init__(self):
        self.CONNECTION_MUTATION_PROBABILITY = 0.8
        self.PERTURBATION_PROBABILITY = 0.9
        self.NEW_NODE_PROBABILIY = 0.1#0.03
        self.NEW_CONNECTION_PROBABILITY = 0.2#0.05
        self.STEP = 0.5
        
        self.INNOVATION_NUMBER = 8
        self.NODE_ID = 6

env = gym.make('CartPole-v0')

pop_size = 4
num_iter = 10
c1 = 1.0
c2 = 1.0
c3 = 0.4
compatibility_threshold = 3.0
crossover_probability = 0.75
disable_probability = 0.75

input_ids = [0, 1, 2, 3]
output_ids = [4, 5]

actions = {}
actions[output_ids[0]] = 0
actions[output_ids[1]] = 1

population = Population(pop_size, input_ids, output_ids)
for i in range(num_iter):
    print(population.evaluate_fitness(env, actions))
    population.speciate(c1, c2, c3, compatibility_threshold)
    population.breed_new_generation(crossover_probability, disable_probability, Config())

env.render(close=True)