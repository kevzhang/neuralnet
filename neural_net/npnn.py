import random
from pathos.multiprocessing import Pool
import numpy as np

def SIGMOID(p, s):
    def g(a):
        # asymptotes at -1 and +1 without offset
        return 1.0 / (1.0 + np.exp(-a / p)) + s
    return g

def compute_bias_gradient(args):
    neural_net = args['neural_net']
    (layer_idx, i) = args['bias_idx']

    old_bias = neural_net.get_bias(layer_idx, i)
    new_bias = old_bias + args['step'] / 128.0
    neural_net.set_bias(layer_idx, i, new_bias)
    squared_error = neural_net.get_training_data_squared_error(args['training_inputs'], args['training_outputs'])
    # only matters for non-mp execution
    neural_net.set_bias(layer_idx, i, old_bias)
    return squared_error

def compute_weight_gradient(args):
    neural_net = args['neural_net']
    (layer_idx, row, col) = args['connection_idx']

    old_weight = neural_net.get_weight(layer_idx, row, col)
    new_weight = old_weight + args['step'] / 128.0
    neural_net.set_weight(layer_idx, row, col, new_weight)
    squared_error = neural_net.get_training_data_squared_error(args['training_inputs'], args['training_outputs'])
    # only matters for non-mp execution
    neural_net.set_weight(layer_idx, row, col, old_weight)
    return squared_error

def random_weight_provider():
    return random.uniform(-1.0, 1.0)

class NeuralNet(object):
    """Fully connected NN"""

    class Params(object):
        def __init__(self, num_inputs, num_outputs, hidden_dimensions, sigmoid_p=1.0, sigmoid_s=0.0,
                weight_provider=random_weight_provider, bias_provider=random_weight_provider):
            self.num_inputs = num_inputs
            self.num_outputs = num_outputs
            self.hidden_dimensions = hidden_dimensions
            self.sigmoid_p = sigmoid_p
            self.sigmoid_s = sigmoid_s
            self.weight_provider = weight_provider
            self.bias_provider = bias_provider

    @staticmethod     
    def from_params(params):
        return NeuralNet(params.num_inputs, params.num_outputs, params.hidden_dimensions, params.sigmoid_p,
                params.sigmoid_s, params.weight_provider, params.bias_provider)

    def __init__(self, num_inputs, num_outputs, hidden_dimensions, sigmoid_p=1.0, sigmoid_s=0.0,
            weight_provider=random_weight_provider, bias_provider=random_weight_provider):
        # Weights / Biases
        self.input_layer_biases = None
        self.input_layer_size = num_inputs

        """array of np arrays"""
        self.hidden_layers_weights = []
        self.hidden_layers_biases = []

        self.output_layer_weights = None
        self.output_layer_biases = None
        self.output_layer_size = num_outputs

        # Functions
        self.sigmoid = SIGMOID(sigmoid_p, sigmoid_s)
        self.weight_provider = weight_provider
        self.bias_provider = bias_provider

        ## Initialize Weights and Biases
        self.__initialize_weights(hidden_dimensions)
        self.__initialize_biases()
        self.__validate_structure()

    def __initialize_weights(self, hidden_dimensions):
        prev_layer_size = self.input_layer_size
        if (hidden_dimensions):
            for hidden_layer_size in hidden_dimensions:
                rows = prev_layer_size
                cols = hidden_layer_size
                layer_weights = np.zeros([rows, cols])
                for r in range(rows):
                    for c in range(cols):
                        layer_weights[r, c] = self.weight_provider()
                prev_layer_size = hidden_layer_size
                self.hidden_layers_weights.append(layer_weights)
        rows = prev_layer_size
        cols = self.output_layer_size
        layer_weights = np.zeros([rows, cols])
        for r in range(rows):
            for c in range(cols):
                layer_weights[r, c] = self.weight_provider()
        self.output_layer_weights = layer_weights

    def __initialize_biases(self):
        self.input_layer_biases = np.zeros([1, self.input_layer_size])
        for i in range(self.input_layer_size):
            self.input_layer_biases[0, i] = self.bias_provider()
        for hidden_layer_weights in self.hidden_layers_weights:
            layer_size = hidden_layer_weights.shape[1]
            layer_bias = np.zeros([1, layer_size])
            for i in range(layer_size):
                layer_bias[0, i] = self.bias_provider()
            self.hidden_layers_biases.append(layer_bias)
        self.output_layer_biases = np.zeros([1, self.output_layer_size])
        for i in range(self.output_layer_size):
            self.output_layer_biases[0, i] = self.bias_provider()

    def __validate_structure(self):
        assert (1, self.input_layer_size) == self.input_layer_biases.shape
        prev_size = self.input_layer_size
        if (self.hidden_layers_weights):
            assert len(self.hidden_layers_weights) == len(self.hidden_layers_biases)
            for hidden_idx in range(len(self.hidden_layers_weights)):
                weight_shape = self.hidden_layers_weights[hidden_idx].shape
                bias_shape = self.hidden_layers_biases[hidden_idx].shape
                assert prev_size == weight_shape[0]
                assert bias_shape == (1, weight_shape[1])
                prev_size = weight_shape[1]
        output_weight_shape = self.output_layer_weights.shape
        output_bias_shape = self.output_layer_biases.shape
        assert prev_size == output_weight_shape[0]
        assert output_bias_shape == (1, output_weight_shape[1])
        assert self.output_layer_size == output_weight_shape[1]

    def intake(self, inputs):
        self.__validate_inputs(inputs)
        forward_mat = self.sigmoid(np.array(inputs) + self.input_layer_biases)
        if (self.hidden_layers_weights):
            for layer_idx in range(len(self.hidden_layers_weights)):
                hidden_layer_weights = self.hidden_layers_weights[layer_idx]
                hidden_layer_biases = self.hidden_layers_biases[layer_idx]
                forward_mat = self.sigmoid(forward_mat.dot(hidden_layer_weights) + hidden_layer_biases)
        forward_mat = forward_mat.dot(self.output_layer_weights) + self.output_layer_biases
        return forward_mat.tolist()

    def get_training_data_squared_error(self, inputs, expected_outputs):
        actual_outputs = self.intake(inputs)
        return NeuralNet.squared_error(expected_outputs, actual_outputs)

    def train_bias(self, training_data, step=0.01):
        assert isinstance(training_data, list)
        training_inputs = [inpt for (inpt, expected) in training_data]
        training_outputs = [expected for (inpt, expected) in training_data]
        self.__validate_inputs(training_inputs)
        self.__validate_expected_outputs(training_outputs)

        initial_squared_error = self.get_training_data_squared_error(training_inputs, training_outputs)

        ##### GRADIENT DESCENT ON BIASES #####
        bias_params = []
        num_hidden_layers = len(self.hidden_layers_weights)
        for layer_idx in range(num_hidden_layers + 1):
            (_, layer_biases) = self.hidden_layers_biases[layer_idx].shape\
                if layer_idx < num_hidden_layers else self.output_layer_biases.shape
            for i in range(layer_biases):
                bias_params.append({
                    'neural_net': self,
                    'bias_idx': (layer_idx, i),
                    'step': step,
                    'training_inputs': training_inputs,
                    'training_outputs': training_outputs
                })

        bias_gradients = [initial_squared_error - sq_err for sq_err in map(compute_bias_gradient, bias_params)]

        # weight vector advances by norm == step
        cur_norm = np.sqrt(sum([x ** 2 for x in bias_gradients]))
        if (cur_norm == 0):
            return initial_squared_error
        adjustment = step / cur_norm

        for bias_idx in range(len(bias_params)):
            bias_param = bias_params[bias_idx] 
            (layer_idx, i) = bias_param['bias_idx']
            new_bias = self.get_bias(layer_idx, i) + bias_gradients[bias_idx] * adjustment
            self.set_bias(layer_idx, i, new_bias)
        after_bias_gradient_error = self.get_training_data_squared_error(training_inputs, training_outputs)
        return after_bias_gradient_error

    def train_weight(self, training_data, step=0.01):
        assert isinstance(training_data, list)
        training_inputs = [inpt for (inpt, expected) in training_data]
        training_outputs = [expected for (inpt, expected) in training_data]
        self.__validate_inputs(training_inputs)
        self.__validate_expected_outputs(training_outputs)

        initial_squared_error = self.get_training_data_squared_error(training_inputs, training_outputs)

        ##### GRADIENT DESCENT ON WEIGHTS #####
        weight_params = []
        num_hidden_layers = len(self.hidden_layers_weights)
        for layer_idx in range(num_hidden_layers + 1):
            (layer_rows, layer_cols) = self.hidden_layers_weights[layer_idx].shape\
                if layer_idx < num_hidden_layers else self.output_layer_weights.shape
            for r in range(layer_rows):
                for c in range(layer_cols):
                    weight_params.append({
                        'neural_net': self,
                        'connection_idx': (layer_idx, r, c),
                        'step': step,
                        'training_inputs': training_inputs,
                        'training_outputs': training_outputs
                    })

        if len(weight_params) < 1000:
            weight_gradients = [initial_squared_error - sq_err for sq_err in map(compute_weight_gradient, weight_params)]
        else:
            weight_gradients = [initial_squared_error - sq_err for sq_err in pool.map(compute_weight_gradient, weight_params)]

        # weight vector advances by norm == step
        cur_norm = np.sqrt(sum([x ** 2 for x in weight_gradients]))
        if (cur_norm == 0):
            return initial_squared_error
        adjustment = step / cur_norm

        for weight_idx in range(len(weight_params)):
            weight_param = weight_params[weight_idx] 
            (layer_idx, row, col) = weight_param['connection_idx']
            new_weight = self.get_weight(layer_idx, row, col) + weight_gradients[weight_idx] * adjustment
            self.set_weight(layer_idx, row, col, new_weight)
        after_edge_gradient_error = self.get_training_data_squared_error(training_inputs, training_outputs)
        return after_edge_gradient_error

    def get_bias(self, layer_idx, i):
        num_hidden_layers = len(self.hidden_layers_weights)
        layer = self.hidden_layers_biases[layer_idx]\
                if layer_idx < num_hidden_layers else self.output_layer_biases
        return layer[0, i]

    def set_bias(self, layer_idx, i, bias):
        num_hidden_layers = len(self.hidden_layers_weights)
        layer = self.hidden_layers_biases[layer_idx]\
                if layer_idx < num_hidden_layers else self.output_layer_biases
        layer[0, i] = bias

    def get_weight(self, layer_idx, row, col):
        num_hidden_layers = len(self.hidden_layers_weights)
        layer = self.hidden_layers_weights[layer_idx]\
                if layer_idx < num_hidden_layers else self.output_layer_weights
        return layer[row, col]

    def set_weight(self, layer_idx, row, col, weight):
        num_hidden_layers = len(self.hidden_layers_weights)
        layer = self.hidden_layers_weights[layer_idx]\
                if layer_idx < num_hidden_layers else self.output_layer_weights
        layer[row, col] = weight

    def __validate_inputs(self, inputs):
        assert isinstance(inputs, list)
        assert len(inputs) > 0
        for inpt in inputs:
            assert isinstance(inpt, list)
            assert len(inpt) == self.input_layer_size,\
                'expecting {0} inputs but got {1}'.format(self.input_layer_size, len(inpt))

    def __validate_expected_outputs(self, outputs):
        assert isinstance(outputs, list)
        assert len(outputs) > 0
        for out in outputs:
            assert isinstance(out, list)
            assert len(out) == self.output_layer_size,\
                'expecting {0} outputs but got {1}'.format(self.output_layer_size, len(out))

    @staticmethod
    def error(expected, actual):
        assert len(expected) == len(actual)
        return np.sqrt(sum([(expected - actual) ** 2 for (expected, actual) in zip(expected, actual)]))

    @staticmethod
    def squared_error(expected_outputs, actual_outputs):
        assert len(expected_outputs) == len(actual_outputs)
        return sum([NeuralNet.error(expected, actual) ** 2\
            for (expected, actual) in zip(expected_outputs, actual_outputs)])

    def __str__(self):
        res = ''
        res += '--- input ---\n'
        res += str(self.input_layer_biases) + '\n'
        res += '--- hidden ---\n'
        for i in range(len(self.hidden_layers_weights)):
            res += str(self.hidden_layers_weights[i]) + '\n'
            res += str(self.hidden_layers_biases[i]) + '\n'
        res += '--- output ---\n'
        res += str(self.output_layer_weights) + '\n'
        res += str(self.output_layer_biases) + '\n'
        res += '--- --- ---'
        return res

pool = Pool(6)