import numpy as np
import torch
import torch.nn as nn
from utils.network_utils import device, np2torch
from submission.mlp import build_mlp


class BaselineNetwork(nn.Module):
    """
    Class for implementing Baseline network

    Args:
        env (): OpenAI gym environment
        config (dict): A dictionary containing generated from reading a yaml configuration file

    TODO:
        Create self.network using build_mlp, and create self.optimizer to
        optimize its parameters.
        You should make use of some values from config, such as the number of layers,
        the size of the layers, and the learning rate.
    """
    def __init__(self, env, config):
        super().__init__()
        self.config = config
        self.env = env
        self.baseline = None
        self.lr = self.config["hyper_params"]["learning_rate"]

        ### START CODE HERE ###
        ### END CODE HERE ###


    def forward(self, observations):
        """
        Pytorch forward method used to perform a forward pass of inputs(observations) 
        through the network

        Args:
            observations (torch.Tensor): observation of state from the environment 
                                        (shape [batch size, dim(observation space)])

        Returns:
            output (torch.Tensor): networks predicted baseline value for a given observation
                                (shape [batch size])

        TODO:
            Run the network forward and then squeeze the result so that it's
            1-dimensional. Put the squeezed result in a variable called "output"
            (which will be returned).

        Note:
            A nn.Module's forward method will be invoked if you
            call it like a function, e.g. self(x) will call self.forward(x).
            When implementing other methods, you should use this instead of
            directly referencing the network (so that the shape is correct).


        """
        ### START CODE HERE ###
        ### END CODE HERE ###
        assert output.ndim == 1
        return output

    def calculate_advantage(self, returns, observations):
        """
        

        Args:
            returns (np.array): the history of discounted future returns for each step (shape [batch size])
            observations (np.array): observations at each step (shape [batch size, dim(observation space)])

        Returns:
            advantages (np.array): returns - baseline values  (shape [batch size])

        TODO:
            Evaluate the baseline and use the result to compute the advantages.
            Put the advantages in a variable called "advantages" (which will be
            returned).

        Note:
            The arguments and return value are numpy arrays. The np2torch function
            converts numpy arrays to torch tensors. You will have to convert the
            network output back to numpy, which can be done via the numpy() method.
            See Converting torch Tensor to numpy Array section of the following tutorial
            for further details: https://pytorch.org/tutorials/beginner/former_torchies/tensor_tutorial.html
            Before converting to numpy, take into consideration the current device of the tensor and whether
            this can be directly converted to a numpy array. Further details can be found here:
            https://pytorch.org/docs/stable/generated/torch.Tensor.cpu.html
        """
        observations = np2torch(observations)
        ### START CODE HERE ###
        ### END CODE HERE ###
        return advantages

    def update_baseline(self, returns, observations):
        """
        Performs back propagation to update the weights of the baseline network according to MSE loss

        Args:
            returns (np.array): the history of discounted future returns for each step (shape [batch size]) 
            observations (np.array): observations at each step (shape [batch size, dim(observation space)])

        TODO:
            Compute the loss (MSE), backpropagate, and step self.optimizer.
            You may (though not necessary) find it useful to do perform these steps
            more than one once, since this method is only called once per policy update.
            If you want to use mini-batch SGD, we have provided a helper function
            called batch_iterator (implemented in utils/network_utils.py).
        """
        returns = np2torch(returns)
        observations = np2torch(observations)
        ### START CODE HERE ###
        ### END CODE HERE ###
