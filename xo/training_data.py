import copy, random
from neural_net.npnn import NeuralNet

width = 8
height = 8

nn_params = NeuralNet.Params(width * height, 1, [16, 1], sigmoid_p=1.0, sigmoid_s=0.0)

def get_random_image():
    img = []
    for _ in range(height):
        img.append(''.join(['x' if random.randint(0, 1) else '.' for _ in range(width)]))
    return img

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
        'x....',
        '.x.x.',
        '..x..',
        '.x.x.',
        'x...x'
    ],
    True
))
training_data.extend(duplicate_and_shift(
    [
        '....x',
        '.x.x.',
        '..x..',
        '.x.x.',
        'x...x'
    ],
    True
))
training_data.extend(duplicate_and_shift(
    [
        'x...x',
        '.x.x.',
        '..x..',
        '.x.x.',
        '....x'
    ],
    True
))
training_data.extend(duplicate_and_shift(
    [
        'x...x',
        '.x.x.',
        '..x..',
        '.x.x.',
        'x....'
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
        'x....x.',
        '.x...x.',
        '..x.x..',
        '..xx...',
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
training_data.extend(duplicate_and_shift(
    [
        '..x..',
        '..x..',
        '..x..',
        'xxxxx',
        '..x..',
        '..x..',
        '..x..'
    ],
    False
))

num_positive_examples = len([x[1] for x in training_data if x[1]])
num_negative_examples = len([x[1] for x in training_data if not x[1]])

training_data.extend([(get_random_image(), False) for _ in range(num_positive_examples - num_negative_examples)])
