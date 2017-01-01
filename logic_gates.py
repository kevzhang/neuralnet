from neural_net.nn import Neuron, NeuralNet, SIGMOID
from neural_net.trainer import train_until
import random

random.seed(12)

def to_digit_list(string):
    return [int(ch) for ch in string]

def to_digit_string(number, num_digits):
    return ('{0:0' + str(num_digits) + 'b}').format(number)

squared_errors = []
# 16 different trainings
for gate in range(0, 16):
    # defines the gate
    answers = (to_digit_list(to_digit_string(gate, 4)))

    training_data = []
    for i in range(4):

        # defines the answer
        answer = answers[i]

        # defines the input
        inputs = to_digit_list(to_digit_string(i, 2))

        training_data.append((inputs, [answer]))

    print training_data
    nn_params = NeuralNet.Params(2, 1, [1], sigmoid_p=0.2, sigmoid_s=0)

    (nn, err) = train_until(nn_params, training_data, initial_step=0.1, threshold=0.01, repetitions=6)

    squared_errors.append((to_digit_string(gate, 4), nn.get_training_data_squared_error(training_data) / len(training_data)))

    for i in range(4):
        # defines the input
        inputs = to_digit_list(to_digit_string(i, 2))

        print inputs, '=', nn.intake(inputs)
    print nn


print 'avg_squared_errors'
for (gate, err) in squared_errors:
    print gate, err
