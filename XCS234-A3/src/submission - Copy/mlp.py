import torch
import torch.nn as nn


def build_mlp(
          input_size,
          output_size,
          n_layers,
          size):
    """
    Builds a multi-layer perceptron in Pytorch based on a user's input

    Args:
        input_size (int): the dimension of inputs to be given to the network
        output_size (int): the dimension of the output
        n_layers (int): the number of hidden layers of the network
        size (int): the size of each hidden layer
    Returns:
        An instance of (a subclass of) nn.Module representing the network.

    TODO:
        Build a feed-forward network (multi-layer perceptron, or mlp) that maps
        input_size-dimensional vectors to output_size-dimensional vectors.
        It should have 'n_layers' hidden layers, each of 'size' units and followed
        by a ReLU nonlinearity. The final layer should be linear (no ReLU).

        As part of your implementation please make use of the following Pytorch 
        functionalities:
        nn.Linear (https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
        nn.Sequential (https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)
    """
    ### START CODE HERE ###
    ### END CODE HERE ###
    