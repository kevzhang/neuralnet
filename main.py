from nn import Neuron, NeuralNet, SIGMOID
from trainer import train_until

nn = NeuralNet(8, 5, [])
print nn

def to_digit_list(string):
    return [int(ch) for ch in string]

def to_digit_string(number, num_digits):
    return ('{0:0' + str(num_digits) + 'b}').format(number)

training_data = []
for i in range(0, 16, 1):
    for j in range(0, 16, 1):
        left_digits = to_digit_list(to_digit_string(i, 4))
        right_digits = to_digit_list(to_digit_string(j, 4))
        expected_digits = to_digit_list(to_digit_string(i + j, 5))
        training_data.append((left_digits + right_digits, expected_digits))

train_until(nn, training_data, initial_step=0.5)

print nn

for i in range(0, 16, 1):
    for j in range(0, 16, 1):
        left_digits = to_digit_list(to_digit_string(i, 4))
        right_digits = to_digit_list(to_digit_string(j, 4))
        print i, '+', j, '=', str(nn.intake(left_digits + right_digits))
