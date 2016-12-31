import math, copy, random
from pathos.multiprocessing import Pool

def SIGMOID(p, s):
    def g(a):
        # asymptotes at -1 and +1 without offset
        return 1.0 / (1.0 + math.e ** (-a / p)) + s
    return g

class Connection(object):
    target = None
    weight = None
    def __init__(self, target, weight):
        assert isinstance(target, Neuron)
        assert isinstance(weight, int) or isinstance(weight, float)
        self.target = target
        self.weight = weight

class Neuron(object):
    activation = 0
    bias = 0
    """List of outbound Connections"""
    connections = []
    sigmoid = None

    def __init__(self, connections, sig, bias=0):
        assert isinstance(connections, list)
        self.connections = connections
        self.sigmoid = sig
        self.activation = bias
        self.bias = bias

    def get_activation(self):
        return self.activation

    def get_connections(self):
        return self.connections

    def propagate(self):
        self_output = self.sigmoid(self.activation)
        for connection in self.connections:
            weighted_output = self_output * connection.weight
            connection.target.accept_activation(weighted_output)

    def accept_activation(self, activation):
        self.activation += activation

    def reset(self):
        self.activation = self.bias

    def __str__(self):
        return '{' + str(self.activation) + '-' + str([str(con.weight) for con in self.connections]) + '}'

def compute_bias_gradient(args):
    neural_net = args['neural_net']
    idx = args['neuron_idx']
    step = args['step']
    training_data = args['training_data']

    neuron = neural_net.get_non_input_neurons()[idx]
    initial_bias = neuron.bias
    # each weight advances step / 128
    neuron.bias += step / 128.0
    squared_error = neural_net.get_training_data_squared_error(training_data)
    neuron.bias = initial_bias
    return squared_error

def compute_connection_gradient(args):
    neural_net = args['neural_net']
    idx = args['connection_idx']
    step = args['step']
    training_data = args['training_data']

    connection = neural_net.get_connections()[idx]
    initial_weight = connection.weight
    # each weight advances step / 128
    connection.weight += step / 128.0
    squared_error = neural_net.get_training_data_squared_error(training_data)
    connection.weight = initial_weight
    return squared_error

def random_weight_provider():
    return random.uniform(-1.0, 1.0)

class NeuralNet(object):
    """Fully connected for now"""
    input_layer = []
    output_layer = []
    hidden_layers = []
    sigmoid = None
    weight_provider = None
    bias_provider = None

    def __init__(self, num_inputs, num_outputs, hidden_dimensions, sigmoid_p=1.0, sigmoid_s=0.0,
            weight_provider=random_weight_provider, bias_provider=random_weight_provider):
        self.sigmoid = SIGMOID(sigmoid_p, sigmoid_s)
        self.weight_provider = weight_provider
        self.bias_provider = bias_provider
        self.output_layer = self.__init_layer(num_outputs, [])
        num_hidden_layers = len(hidden_dimensions)
        hidden_layers = [None] * num_hidden_layers
        for i in range(num_hidden_layers - 1, -1, -1):
            if i == num_hidden_layers - 1:
                outbound_layer = self.output_layer
            else:
                outbound_layer = hidden_layers[i + 1]
            hidden_layers[i] = self.__init_layer(hidden_dimensions[i], outbound_layer) 
        self.hidden_layers = hidden_layers
        self.input_layer = self.__init_layer(num_inputs, hidden_layers[0] if num_hidden_layers else self.output_layer)

    def __init_layer(self, layer_size, outbound_layer):
        layer = [None] * layer_size
        for i in xrange(layer_size):
            # Weights default to 1
            layer[i] = Neuron([Connection(neuron, self.weight_provider()) for neuron in outbound_layer], self.sigmoid, bias=self.bias_provider())
        return layer

    def __reset(self):
        for neuron in self.input_layer:
            neuron.reset()
        for layer in self.hidden_layers:
            for neuron in layer:
                neuron.reset()
        for neuron in self.output_layer:
            neuron.reset()

    def __propagate_all_layers(self):
        for neuron in self.input_layer:
            neuron.propagate()
        for layer in self.hidden_layers:
            for neuron in layer:
                neuron.propagate()

    def __set_inputs(self, inputs):
        for i in xrange(len(inputs)):
            neuron = self.input_layer[i]
            activation = inputs[i]
            neuron.accept_activation(activation)

    def __intake(self, inputs):
        self.__set_inputs(inputs)
        self.__propagate_all_layers()

    def intake(self, inputs):
        self.__validate_inputs(inputs)
        self.__reset()

        self.__intake(inputs)
        output = [neuron.get_activation() for neuron in self.output_layer]

        self.__reset()
        return output

    def get_training_data_squared_error(self, training_data):
        expected_and_actual_list = [(expected_outputs, self.intake(inputs)) for (inputs, expected_outputs) in training_data]
        return NeuralNet.squared_error(expected_and_actual_list)

    def copy_self(self):
        return copy.deepcopy(self)

    def train_weight(self, training_data, step=0.01):
        assert isinstance(training_data, list)
        for (inputs, expected_outputs) in training_data:
            self.__validate_inputs(inputs)
            self.__validate_expected_outputs(expected_outputs)

        initial_squared_error = self.get_training_data_squared_error(training_data)

        ##### GRADIENT DESCENT ON WEIGHTS #####
        connections = self.get_connections()
        num_connections = len(connections)
        worker_args = []
        for idx in range(num_connections):
            worker_args.append({
                'neural_net': self,
                'connection_idx': idx,
                'step': step,
                'training_data': training_data
            })

        connection_gradients = [initial_squared_error - sq_err for sq_err in pool.map(compute_connection_gradient, worker_args)]

        # weight vector advances by norm == step
        cur_norm = math.sqrt(sum([x ** 2 for x in connection_gradients]))
        if (cur_norm == 0):
            return initial_squared_error
        adjustment = step / cur_norm

        prev_weights = [connections[i].weight for i in range(num_connections)]
        for i in xrange(num_connections):
            connections[i].weight += connection_gradients[i] * adjustment
        after_edge_gradient_error = self.get_training_data_squared_error(training_data)
        if (after_edge_gradient_error >= initial_squared_error):
            print 'Resetting weights due to regression'
            for i in xrange(num_connections):
                connections[i].weight = prev_weights[i]
            return initial_squared_error
        else:
            return after_edge_gradient_error

    def train_bias(self, training_data, step=0.01):
        assert isinstance(training_data, list)
        for (inputs, expected_outputs) in training_data:
            self.__validate_inputs(inputs)
            self.__validate_expected_outputs(expected_outputs)

        initial_squared_error = self.get_training_data_squared_error(training_data)

        ##### GRADIENT DESCENT ON BIAS #####
        neurons = self.get_non_input_neurons()
        num_neurons = len(neurons)
        worker_args = []
        for idx in range(num_neurons):
            worker_args.append({
                'neural_net': self,
                'neuron_idx': idx,
                'step': step,
                'training_data': training_data
            })

        bias_gradients = [initial_squared_error - sq_err for sq_err in pool.map(compute_bias_gradient, worker_args)]

        # weight vector advances by norm == step
        cur_norm = math.sqrt(sum([x ** 2 for x in bias_gradients]))
        if (cur_norm == 0):
            return initial_squared_error
        adjustment = step / cur_norm

        prev_bias = [neurons[i].bias for i in xrange(num_neurons)]
        for i in xrange(num_neurons):
            neurons[i].bias += bias_gradients[i] * adjustment
        after_bias_gradient_error = self.get_training_data_squared_error(training_data)
        if (after_bias_gradient_error >= initial_squared_error):
            print 'Resetting bias due to regression'
            for i in xrange(num_neurons):
                neurons[i].bias = prev_bias[i]
            return initial_squared_error
        else:
            return after_bias_gradient_error

    def get_non_input_neurons(self):
        neurons = []
        for layer in self.hidden_layers:
            for neuron in layer:
                neurons.append(neuron)
        for neuron in self.output_layer:
            neurons.append(neuron)
        return neurons

    def get_connections(self):
        connections = []
        for neuron in self.input_layer:
            connections.extend(neuron.get_connections())
        for layer in self.hidden_layers:
            for neuron in layer:
                connections.extend(neuron.get_connections())
        return connections

    def __validate_inputs(self, inputs):
        assert len(inputs) == len(self.input_layer),\
            'expecting {0} inputs but got {1}'.format(len(self.input_layer), len(inputs))

    def __validate_expected_outputs(self, outputs):
        assert len(outputs) == len(self.output_layer),\
            'expecting {0} outputs but got {1}'.format(len(self.output_layer), len(outputs))

    @staticmethod
    def error(expected_outputs, actual_outputs):
        assert len(expected_outputs) == len(actual_outputs)
        return math.sqrt(sum([(expected - actual) ** 2 for (expected, actual) in zip(expected_outputs, actual_outputs)]))

    @staticmethod
    def squared_error(expected_and_actual_list):
        return sum([NeuralNet.error(expected_outputs, actual_outputs) ** 2\
            for (expected_outputs, actual_outputs) in expected_and_actual_list])

    def __str__(self):
        string = 'input ({0}): '.format(len(self.input_layer))
        for neuron in self.input_layer:
            string += str(neuron)
        string += '\n'
        string += 'hidden with {0} layers: '.format(len(self.hidden_layers))
        for layer in self.hidden_layers:
            string += '\n    '
            for neuron in layer:
                string += str(neuron)
        string += '\n'
        string += 'output ({0}): '.format(len(self.output_layer))
        for neuron in self.output_layer:
            string += str(neuron)
        string += '\n'
        return string

pool = Pool(4)