from neuron import Neuron

class Edge:

    def __init__(self, head: Neuron, tail: Neuron, weight) -> None:
        self.weight = weight
        self.head = head
        self.tail = tail