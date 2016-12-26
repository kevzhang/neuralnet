from nn import Neuron, NeuralNet, SIGMOID

nn = NeuralNet(4, 1, [])
print nn

training_data = []
for i in range(16):
    digits = [int(c) for c in '{0:04b}'.format(i)]
    # digits = [-1 if d == 0 else 1 for d in digits]
    training_data.append((digits, [i]))

for x in xrange(1000):
    nn.train(training_data, step=0.1)

print nn

for i in range(16):
    digits = [int(c) for c in '{0:04b}'.format(i)]
    # digits = [-1 if d == 0 else 1 for d in digits]
    print '{0:04b}'.format(i) + ' => ' + str(nn.intake(digits))

print SIGMOID(1)(0)