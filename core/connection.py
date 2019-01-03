class Connection:
	def __init__(self, innovation_number, from_key, to_key, weight, enabled):
		self.innovation_number = innovation_number
		self.from_key = from_key
		self.to_key = to_key
		self.weight = weight
		self.enabled = enabled
