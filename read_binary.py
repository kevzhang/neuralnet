from neural_net.nn import Neuron, NeuralNet, SIGMOID
from neural_net.trainer import train_until

#works well!
nn = NeuralNet(4, 1, [], sigmoid_p=0.01, sigmoid_s=-0.5)
print nn

def to_digit_list(string):
    return [int(ch) for ch in string]

def to_digit_string(number, num_digits):
    return ('{0:0' + str(num_digits) + 'b}').format(number)

training_data = []
for i in range(0, 16, 1):
    digits = to_digit_list(to_digit_string(i, 4))
    training_data.append((digits, [i]))

train_until(nn, training_data, initial_step=0.5, threshold=0.01)

print nn

for i in range(0, 16, 1):
    digits = to_digit_list(to_digit_string(i, 4))
    print digits, '=', str(nn.intake(digits))
