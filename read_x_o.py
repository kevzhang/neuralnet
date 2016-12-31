from neural_net.nn import Neuron, NeuralNet, SIGMOID
from neural_net.trainer import train_until
import copy, random

random.seed(123)

width = 8
height = 8

x_val = 1
o_val = 0

true_val = 1
false_val = 0

nn = NeuralNet(width * height, 1, [9], sigmoid_s=0.0)

def to_nn_data(raw_training_data):
    return [(image_to_input(image), [true_val] if result else [false_val]) for (image, result) in raw_training_data]

def duplicate_and_shift(image, result):
    img_height = len(image)
    img_width = len(image[0])
    heightD = height - img_height
    widthD = width - img_width
    shifted = []
    for row_offset in range(heightD + 1):
        for col_offset in range(widthD + 1):
            right_cols = width - img_width - col_offset
            bot_rows = height - img_height - row_offset
            shifted_image = copy.copy(image)
            # fill with left and right columns
            for imgR in range(len(shifted_image)):
                shifted_image[imgR] = col_offset * '.' + shifted_image[imgR] + right_cols * '.'
            shifted_image = row_offset * ['.' * width] + shifted_image + bot_rows * ['.' * width]
            shifted.append((shifted_image, result))
    return shifted

def get_random_image():
    img = []
    for _ in range(height):
        img.append(''.join(['x' if random.randint(0, 1) else '.' for _ in range(width)]))
    return img

training_data = []
training_data.extend(duplicate_and_shift(
    [
        'x.x',
        '.x.',
        'x.x'
    ],
    True
))
training_data.extend(duplicate_and_shift(
    [
        'x...x',
        '.x.x.',
        '..x..',
        '.x.x.',
        'x...x'
    ],
    True
))
training_data.extend(duplicate_and_shift(
    [
        'x.....x',
        '.x...x.',
        '..x.x..',
        '...x...',
        '..x.x..',
        '.x...x.',
        'x.....x'
    ],
    True
))
training_data.extend(duplicate_and_shift(
    [
        'xxxx',
        'xxxx',
        'xxxx',
        'xxxx'
    ],
    False
))
training_data.extend(duplicate_and_shift(
    [
        'xxxxxxxx',
        'xxxxxxxx',
        'xxxxxxxx',
        'xxxxxxxx',
        'xxxxxxxx',
        'xxxxxxxx',
        'xxxxxxxx',
        'xxxxxxxx'
    ],
    False
))
training_data.extend(duplicate_and_shift(
    [
        '........',
        '........',
        '........',
        '........',
        '........',
        '........',
        '........',
        '........'
    ],
    False
))
training_data.extend(duplicate_and_shift(
    [
        '..x..',
        '.x.x.',
        'x...x',
        '.x.x.',
        '..x..'
    ],
    False
))
training_data.extend(duplicate_and_shift(
    [
        'xxxxx',
        'x...x',
        'x...x',
        'x...x',
        'xxxxx'
    ],
    False
))

num_positive_examples = len([x[1] for x in training_data if x[1]])
num_negative_examples = len([x[1] for x in training_data if not x[1]])

training_data.extend([(get_random_image(), False) for _ in range(num_positive_examples - num_negative_examples)])

def image_to_input(image):
    str_img = ''.join(image)
    return [-1.0 if c == '.' else 1.0 for c in str_img]

nn_training_data = to_nn_data(training_data)
print '|training_data|', len(training_data)
train_until(nn, nn_training_data, initial_step=0.5, threshold=0.5)

test_data = [
    (
        [
            'x......x',
            '........',
            '..xxx...',
            '..xxx...',
            '..xxx...',
            '........',
            '........',
            'x......x'
        ],
        False
    ),
    (
        [
            '........',
            '...xxx..',
            '..x...x.',
            '...x.x..',
            '...xx...',
            '...x.x..',
            '..x...x.',
            '..xxxxx.'
        ],
        False
    ),
    (
        [
            '........',
            '...xxx..',
            '..x...x.',
            '.x.....x',
            '.x.....x',
            '..x...x.',
            '...xxx..',
            '........'
        ],
        False
    ),
    (
        [
            '........',
            '.xxxxxx.',
            '.xx...x.',
            '.x.x....',
            '.x......',
            '.x.x.x..',
            '.xx...x.',
            '.xxxxxxx'
        ],
        False
    ),
    (
        [
            '........',
            '........',
            '........',
            '........',
            '........',
            '........',
            '........',
            '........'
        ],
        False
    ),

    (
        [
            'xxxxxxxx',
            'xxxxxxxx',
            'xxxxxxxx',
            'xxxxxxxx',
            'xxxxxxxx',
            'xxxxxxxx',
            'xxxxxxxx',
            'xxxxxxxx'
        ],
        False
    ),
    (
        [
            '........',
            '..xxx...',
            '....x...',
            '..xxx...',
            '....x...',
            '..xxx...',
            '........',
            '........'
        ],
        False
    ),
    (
        [
            '........',
            '..xxxx..',
            '..x.....',
            '..xxxx..',
            '.....x..',
            '.....x..',
            '..xxx...',
            '........'
        ],
        False
    ),
    (
        [
            '...x...x',
            '....x.x.',
            '.....x..',
            '....x.x.',
            '...x...x',
            '........',
            '........',
            '........'
        ],
        True
    ),

    (
        [
            '........',
            '........',
            '..x.....',
            '...x...x',
            '....x.x.',
            '....xx..',
            '...x..x.',
            '..x....x'
        ],
        True
    ),
    (
        [
            '........',
            '........',
            '..x.x...',
            '...x....',
            '..x.x...',
            '........',
            '........',
            '........'
        ],
        True
    ),
    (
        [
            '........',
            '.x...x..',
            '..x.x...',
            '...x....',
            '..x.x...',
            '.x...x..',
            '........',
            '........'
        ],
        True
    ),
    (
        [
            'x.....x.',
            '.x...x..',
            '..x.x...',
            '...x....',
            '..x.....',
            '.x...x..',
            'x.....x.',
            '........'
        ],
        True
    ),
    (
        [
            '........',
            '..x..x..',
            '..x.x...',
            '...x....',
            '..x.x...',
            '.x...x..',
            '........',
            '........'
        ],
        True
    ),
    (
        [
            '........',
            '.x......',
            '..x...x.',
            '...x.x..',
            '...xx...',
            '...x.x..',
            '..x...x.',
            '.x.....x'
        ],
        True
    )
]

for i in range(len(test_data)):
    (image, result) = test_data[i]
    print '\n'.join(image)
    print '\n=\n'
    print nn.intake(image_to_input(image))
    print '\n'

nn_testing_data = to_nn_data(test_data)
print 'avg_test_squared_error', nn.get_training_data_squared_error(nn_testing_data) / len(nn_testing_data)
