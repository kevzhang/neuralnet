from nn import Neuron, NeuralNet, SIGMOID

nn = NeuralNet(3, 3, [3])
print nn

nn.intake([0,0,0])

print nn

print SIGMOID(1)(1.5) * 3

