from train import train_network, test_network
from solvers import *
from utils import generate_disc_set


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
    mod_list = [Linear(n_input, 10, 0.0, 1.0), Relu(),
                Linear(10, 25, 0.0, 1.0), Relu(),
                Linear(25, 25, 0.0, 1.0), Relu(),
                Linear(25, 10, 0.0, 1.0), Relu(),
                Linear(10, n_output, 0.0, 1.0)]
    # Train with MSE Loss
    criterion = MSEloss()

    # # Log the Loss
    net = train_network(mod_list, criterion, train_data, train_labels, lr=1e-5, network_name='mse',
                        epochs=200, batch_size=5)

    # Compute and print final test error
    test_network(net)

    # Example using CrossEntropyLoss

    # mod_list2 = [Linear(n_input, 10, 0.0, 1.0), Tanh(),
    #              Linear(10, 50, 0.0, 1.0), Tanh(),
    #              Linear(50, 50, 0.0, 1.0), Tanh(),
    #              Linear(50, 10, 0.0, 1.0), Relu(),
    #              Linear(10, n_output, 0.0, 1.0), LeakyRelu()]
    # criterion2 = CrossEntropyLoss()
    #
    # net2 = train_network(mod_list2, criterion2, train_data, train_labels, lr=1e-4, network_name='cross_entropy', batch_size=5)
    # test_network(net2)


if __name__ == "__main__":
    main()
