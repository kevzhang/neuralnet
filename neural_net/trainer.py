from neural_net.npnn import NeuralNet

class RollingErrorHistory(object):
    history = []
    history_size = 0

    def __init__(self, history_size):
        self.history_size = history_size

    def __record_error(self, error):
        if (len(self.history) < self.history_size):
            self.history.append(error)
        else:
            self.history = self.history[1:]
            self.history.append(error)

    def record_and_compare(self, error):
        self.__record_error(error)
        
        if (len(self.history) == self.history_size):
            return self.history[0] - self.history[-1]
        else:
            return None

def train_until(neural_net_params, training_data, initial_step=0.1, threshold=0.1, repetitions=1):
    neural_nets = [NeuralNet.from_params(neural_net_params) for _ in range(repetitions)]
    trained = [__train_until(nn, training_data, initial_step, threshold) for nn in neural_nets]
    (best_nn, best_error) = trained[0]
    for (nn, error) in trained[1:]:
        if error < best_error:
            best_nn = nn
            best_error = error
    print 'all errors', [x[1] / len(training_data) for x in trained]
    return (best_nn, best_error)

# Threshold is still with respect to total squared error
def __train_until(neural_net, training_data, initial_step, threshold):
    history = RollingErrorHistory(10)
    num_examples = len(training_data)

    current_weight_step = initial_step
    current_bias_step = initial_step

    training_inputs = [training_input for (training_input, _) in training_data]
    training_outputs = [expected_output for (_, expected_output) in training_data]
    prev_squared_error = neural_net.get_training_data_squared_error(training_inputs, training_outputs)
    history.record_and_compare(prev_squared_error)
    print 'avg_sq_error', prev_squared_error / num_examples

    cycles = 0
    while True:
        cycles += 1
        # Train edge weights
        weight_squared_error = neural_net.train_weight(training_data, step=current_weight_step)
        if weight_squared_error < prev_squared_error:
            current_weight_step *= 1.2
        else:
            current_weight_step /= 4.0
        print 'after_weights', weight_squared_error / num_examples
        # Train neuron bias
        bias_squared_error = neural_net.train_bias(training_data, step=current_bias_step)
        if bias_squared_error < weight_squared_error:
            current_bias_step *= 1.2
        else:
            current_bias_step /= 4.0
        print 'after_bias', bias_squared_error / num_examples

        final_squared_error = bias_squared_error
        progress = history.record_and_compare(final_squared_error)
        if (progress != None and abs(progress) < threshold):
            print 'progress in last 10 rounds = ' + str(progress) + ' - halt training'
            return (neural_net, final_squared_error)
        print 'step_sizes: edge-step', current_weight_step, 'bias-step', current_bias_step
        print 'num_cycles', cycles
        prev_squared_error = final_squared_error


