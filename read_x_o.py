from neural_net.nn import Neuron, NeuralNet, SIGMOID
from neural_net.trainer import train_until

training_data = [
    (
        '........' +
        '.x...x..' + 
        '..x.x...' + 
        '...x....' + 
        '..x.x...' + 
        '.x...x..' + 
        '........' +
        '........',
        True
    ),
    (
        '........' +
        '..x...x.' + 
        '...x.x..' + 
        '....x...' + 
        '...x.x..' + 
        '..x...x.' + 
        '........' +
        '........',
        True
    ),
    (
        '........' +
        '........' +
        '..x...x.' + 
        '...x.x..' + 
        '....x...' + 
        '...x.x..' + 
        '..x...x.' + 
        '........',
        True
    ),
    (
        '........' +
        '........' +
        '.x...x..' + 
        '..x.x...' + 
        '...x....' + 
        '..x.x...' + 
        '.x...x..' + 
        '........',
        True
    ),
    (
        'x.....x.' +
        '.x...x..' + 
        '..x.x...' + 
        '...x....' + 
        '..x.x...' + 
        '.x...x..' + 
        'x.....x.' +
        '........',
        True
    ),
    (
        '.x.....x' +
        '..x...x.' + 
        '...x.x..' + 
        '....x...' + 
        '...x.x..' + 
        '..x...x.' + 
        '.x.....x' +
        '........',
        True
    ),
    (
        '........' +
        '.x.....x' +
        '..x...x.' + 
        '...x.x..' + 
        '....x...' + 
        '...x.x..' + 
        '..x...x.' + 
        '.x.....x',
        True
    ),
    (
        '........' +
        'x......x' +
        '.x...x..' + 
        '..x.x...' + 
        '...x....' + 
        '..x.x...' + 
        '.x...x..' + 
        'x.....x.',
        True
    ),
    (
        '........' +
        '...xx...' +
        '.x...xx.' + 
        'x....x..' + 
        'x.xx..x.' + 
        '.x......' + 
        '..x.x...' + 
        'x.....x.',
        False
    ),
    (
        '.......x' +
        '...xx...' +
        '.xx..xx.' + 
        'x....x..' + 
        'x.xx..x.' + 
        '.x......' + 
        '..x.x...' + 
        '......x.',
        False
    ),
    (
        '........' +
        '..xxx...' +
        '.x...x..' + 
        'xxxxxxx.' + 
        'x.....x.' + 
        '.x...x..' + 
        '..xxx...' + 
        '........',
        False
    ),
    (
        '........' +
        '..xxx...' +
        '.x...x..' + 
        'x.....x.' + 
        'x.....x.' + 
        '.x...x..' + 
        '..xxx...' + 
        '........',
        False
    ),
    (
        '........' +
        'xxx.xxxx' +
        '.xxxxx..' + 
        '..xxx...' + 
        '..xxx...' + 
        '..xxx...' + 
        '.xxxxx..' + 
        'xxx.xxx.',
        False
    )
]

def chunks(l, n):
    for idx in range(0, len(l), n):
        yield l[idx:idx + n]

def image_to_input(image):
    return [-1.0 if c == '.' else 1.0 for c in image]

nn_training_data = [(image_to_input(image), [1] if result else [-1]) for (image, result) in training_data]

#works well!
nn = NeuralNet(64, 1, [16], sigmoid_s=-0.5)

print nn

def to_digit_list(string):
    return [int(ch) for ch in string]

def to_digit_string(number, num_digits):
    return ('{0:0' + str(num_digits) + 'b}').format(number)

train_until(nn, nn_training_data, initial_step=0.5, threshold=0.01)

test_data = [
    (
        '........' +
        '........' + 
        '..x.x...' + 
        '...x....' + 
        '..x.x...' + 
        '........' + 
        '........' +
        '........',
        True
    ),
    (
        '..x.....' +
        '..x.....' + 
        '..x.x...' + 
        '........' + 
        '..x.x...' + 
        '..x.x...' + 
        '..x.x...' +
        '........',
        False
    ),
    (
        '........' +
        '..x...x.' + 
        '...x.x..' + 
        '....x...' + 
        '...x.x..' + 
        '..x...x.' + 
        '........' +
        '........',
        True
    ),
    (
        '........' +
        '........' +
        '..xxxxx.' + 
        '..x...x.' + 
        '..x...x.' + 
        '..x...x.' + 
        '..xxxxx.' + 
        '........',
        False
    ),
    (
        '........' +
        '........' +
        '.x...x..' + 
        '..x.x...' + 
        '...x....' + 
        '..x.x...' + 
        '.x...x..' + 
        '........',
        True
    ),
    (
        '........' +
        '.x......' +
        '.xxxxx..' + 
        '........' + 
        '........' + 
        '..x.xx..' + 
        '.x...x..' + 
        '........',
        True
    )
]

for i in range(len(test_data)):
    (image, result) = test_data[i]
    print '\n'.join(list(chunks(image, 8)))
    print '\n=\n'
    print nn.intake(image_to_input(image))
    print '\n'

nn_testing_data = [(image_to_input(image), [1] if result else [-1]) for (image, result) in test_data]
print 'test_squared_error', nn.get_training_data_squared_error(nn_testing_data)