from Config import config
import networkx as nx
import matplotlib.pyplot as plt


class Interface:
	def __init__(self):
		self.node_positions = {}

		num_inputs = len(config.input_keys)
		for i, key in enumerate(config.input_keys):
			x = 0
			y = (num_inputs - (i + 1)) * config.network_canvas_height / (num_inputs - 1)

			self.node_positions[key] = (x, y)

		num_outputs = len(config.output_keys)
		for i, key in enumerate(config.output_keys):
			x = config.network_canvas_width
			y = (num_outputs - (i + 1)) * config.network_canvas_height / (num_outputs - 1)

			self.node_positions[key] = (x, y)

	def visualize_network(self, connections, nodes):
		edges = [(connection.from_key, connection.to_key, round(connection.weight, 2)) for connection in connections.values() if connection.enabled]

		G = nx.DiGraph()
		G.add_weighted_edges_from(edges)

		# todo: ugly
		for key in nodes.keys():
			if key not in self.node_positions.keys():
				neighbor_nodes = [connection.to_key if connection.from_key == key else connection.from_key for connection in connections.values() if connection.to_key == key or connection.from_key == key]
				x = sum([self.node_positions[node_key][0] for node_key in neighbor_nodes]) / len(neighbor_nodes)
				y = sum([self.node_positions[node_key][1] for node_key in neighbor_nodes]) / len(neighbor_nodes)

				self.node_positions[key] = (x, y)

		nx.draw(G, self.node_positions, with_labels=True)
		labels = nx.get_edge_attributes(G, 'weight')
		nx.draw_networkx_edge_labels(G, self.node_positions, edge_labels=labels)

		plt.show()
