from train import train_network, test_network
from utils import generate_disc_set
from module import *


def main():
    # Generate training and test set of 1000 points
    num_train = 1000
    normalize = True
    one_hot = True
    train_data, train_labels = generate_disc_set(num_train,
                                                 normalize=normalize, one_hot=one_hot)

    # Build a Network with two input units, two output units, 3 hidden layers of 25 units
    n_input = 2
    n_output =2
    mod_list = [Linear(n_input, 25, 0.0, 1.0), Relu(),
                Linear(25, 25, 0.0, 1.0), Relu(),
                Linear(25, 25, 0.0, 1.0), Relu(),
                Linear(25, n_output, 0.0, 1.0)]
    # Train with MSE Loss
    criterion = MSEloss()

    # # Log the Loss
    net = train_network(mod_list, criterion, train_data, train_labels, lr=1e-4, network_name='vanilla_network',
                        epochs=200, batch_size=5)

    # Compute and print final test error
    test_network(net)

    # Example using CrossEntropyLoss

    mod_list2 = [Linear(n_input, 10, 0.0, 1.0), LeakyRelu(),
                 Linear(10, 50, 0.0, 1.0), LeakyRelu(),
                 Linear(50, 100, 0.0, 1.0), LeakyRelu(),
                 Linear(100, 100, 0.0, 1.0), LeakyRelu(),
                 Linear(100, 50, 0.0, 1.0), LeakyRelu(),
                 Linear(50, 10, 0.0, 1.0), LeakyRelu(),
                 Linear(10, n_output, 0.0, 1.0)]
    criterion2 = CrossEntropyLoss()

    net2 = train_network(mod_list2, criterion2, train_data, train_labels, lr=1e-4, network_name='cross_entropy', batch_size=5, epochs=200)
    test_network(net2)


if __name__ == "__main__":
    main()
