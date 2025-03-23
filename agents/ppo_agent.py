"""
This file contains the A2C agent to do PPO
Different blocks of functioning are written as separate functions
"""

import numpy as np
from torch.distributions import Categorical
from torch import nn


def initialize_layer(layer, std = np.sqrt(2.0), bias_const = 0.0):
    """
        Initializes the weights and biases of a given layer.

        Parameters:
        layer (nn.Module): The neural network layer to initialize.
        std (float): The standard deviation for orthogonal initialization of weights. Default is sqrt(2).
        bias_const (float): The constant value to initialize the biases. Default is 0.0.

        Returns:
        nn.Module: The initialized layer.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def build_network(nodes_counts, std = 0.01):
    """
       Constructs a neural network with fully connected layers and Tanh activations based on the specified node counts.

       Parameters:
       nodes_counts (list of int): A list where each element represents the number of nodes in a layer.
       std (float): The standard deviation for initializing the final layer's weights. Default is 0.01.

       Returns:
       layers: A list of layers (including activation functions) representing the neural network.
    """
    layers = [initialize_layer(nn.Linear(nodes_counts[0], nodes_counts[1])), nn.Tanh()]

    for i in range(1, len(nodes_counts) - 2):
        layers.append(initialize_layer(nn.Linear(nodes_counts[i], nodes_counts[i + 1])))
        layers.append(nn.Tanh())

    #append the final layer without activation
    layers.append(initialize_layer(nn.Linear(nodes_counts[-2], nodes_counts[-1]), std=std))

    return layers



#this agent is the same as A. Shehper's, for details, see https://github.com/shehper/AC-Solver/blob/main/ac_solver/agents/ppo_agent.py
class PPOAgent(nn.Module):
    """
    A reinforcement learning agent that includes both a critic network for value estimation
    and an actor network for policy generation.

    Attributes:
    critic (nn.Sequential): The neural network used for value estimation.
    actor (nn.Sequential): The neural network used for policy generation.
    """

    def __init__(self, envs, nodes_counts):
        """
        Initializes the Agent with specified environment and node counts for the neural networks.

        Parameters:
        envs (gym.Env): The environment for which the agent is being created.
        nodes_counts (list of int): A list where each element represents the number of nodes in a hidden layer.
        """
        super(PPOAgent, self).__init__()

        input_dim = np.prod(envs.single_observation_space.shape)
        self.critic_nodes = [input_dim] + nodes_counts + [1]
        self.actor_nodes = [input_dim] + nodes_counts + [envs.single_action_space.n]

        self.critic = nn.Sequential(*build_network(self.critic_nodes, 1.0))
        self.actor = nn.Sequential(*build_network(self.actor_nodes, 0.01))


    def get_value(self, state):
        """
        Computes the value of a given state using the critic network.
        :param state: torch.tensor, The input tensor representing the state.

        :return torch.Tensor: The value of the given state.
        """
        return self.critic(state)

    def get_log_prob(self, states_v, actions_v):
        """
        function to calculate the log_prob to get old_log_prob in ppo training
        :param states_v: torch.FloatTensor
        :param actions_v:torch.FloatTensor
        :return: log_prob tensor pi(a|s)
        """
        logits = self.actor(states_v)
        probs = Categorical(logits=logits)
        return probs.log_prob(actions_v)


    def get_action_and_value(self, state, action = None):
        """
        Computes the action to take and its associated value, log probability, and entropy.

        :param state: torch.tensor, The input tensor representing the state.
        :param action: (torch.tensor, optional) The action to evaluate. If None, a new action will be sampled.

        :returns tuple: A tuple containing the action, its log probability, the entropy of the action distribution, and the value of the state.
        """
        logits = self.actor(state)
        value = self.critic(state)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value