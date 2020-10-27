import numpy as np

from neural_network import NeuralNetwork

hidden_nodes = 100
output_nodes = 10
learning_rate = 0.2


def read(src):
    file = open(src, 'r')
    training_data = file.readlines()
    file.close()
    return training_data


def construct_network(num_nodes):
    input_nodes = num_nodes
    return NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


def scaled_array(inputs):
    return inputs / 255.0 * 0.99 + 0.01


if __name__ == '__main__':
    src = "mnist_dataset/mnist_train.csv"
    training_data = read(src)
    all_training_values = training_data[0].split(',')

    neural_net = construct_network(len(all_training_values) - 1)

    # train the neural network
    print("Start training...")
    for record in training_data:
        all_values = record.split(',')
        inputs = scaled_array(np.asfarray(all_values[1:]))
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        neural_net.train(inputs, targets)
    print("Finish training...")

    test_data = read("mnist_dataset/mnist_test.csv")
    total = len(test_data)
    count = 0
    for i in range(0, total):
        all_test_values = test_data[i].split(",")
        expect = all_test_values[0]
        query_array = scaled_array(np.asfarray(all_test_values[1:]))
        outputs = neural_net.query(query_array)
        actual = np.argmax(outputs)

        if int(expect) == actual:
            count += 1
    print("total=%d correct=%d grade=%f" % (total, count, count / total))
