from neural_net.trainer import train_until
from xo.test_data import test_data
from xo.training_data import training_data
from xo.training_data import nn_params

x_val = 1
o_val = -1

true_val = 1
false_val = 0

def to_nn_data(raw_training_data):
    return [(image_to_input(image), [true_val] if result else [false_val]) for (image, result) in raw_training_data]

def image_to_input(image):
    str_img = ''.join(image)
    return [-1.0 if c == '.' else 1.0 for c in str_img]

nn_training_data = to_nn_data(training_data)

print '|training_data|', len(training_data)
(nn, err) = train_until(nn_params, nn_training_data, initial_step=0.1, threshold=0.1)

for i in range(len(test_data)):
    (image, result) = test_data[i]
    print '\n'.join(image)
    print '\n=\n'
    print nn.intake([image_to_input(image)])
    print '\n'

nn_testing_data = to_nn_data(test_data)
training_inputs = [inpt for (inpt, expected) in nn_testing_data]
training_outputs = [expected for (inpt, expected) in nn_testing_data]
print 'avg_test_squared_error', nn.get_training_data_squared_error(training_inputs, training_outputs) / len(nn_testing_data)

print nn
