"""
This file set up the core training loops for PPO.
Different phases of the training loop are separated as individual functions to allow flexibility of change and improvements.
The keys modules include:
    1. rollout: the rollout phase to get the trajectory. Inputs includes the environments where rollout happens in, and max horizon length.
    2. train: run through the epochs of nn updates.
"""
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from tensorboardX import SummaryWriter
import ptan
import lib
import torch.nn.functional as F

from agents.ppo_agent import PPOAgent
from envs.load_envs import load_all_presentations
from envs.ac_env import ACEnv
from envs.rollout import rollout

import sys


#PPO related hyperparameters
GAMMA = 0.99
GAE_LAMBDA = 0.95
LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 1e-4
NUM_EPOCHS = 5
LR = 2.5e-4
EPS = 0.00001

clip_epsilon = 0.2
value_loss_coef=0.5
entropy_coef=0.01


def make_env(initial_state, max_relator_length):
    """
    Creates an environment initialization function (thunk) with the specified configuration.
    :param initial_state: the initial state of the environment
    :param max_relator_length: the maximum relator length
    :return: a thunk to create a new env instance
    """
    #TODO: give more explanation

    def thunk():

        return ACEnv(initial_state, max_relator_length)

    return thunk

def get_envs(all_presentations, indices, max_relator_length):
    """
    the function to get the gym vectorized envs with corresponding indices.
    :param all_presentations: the list of all presentations we want to choose in
    :param indices: the list of indices of the desired presentations to rollout.
    :param max_relator_length: the maximum relator length
    :return: gym vectorized envs.
    """
    envs = gym.vector.SyncVectorEnv(
        [make_env(all_presentations[i], max_relator_length) for i in indices]
    )
    return envs


def get_advantage_and_reference(rewards, dones, values, device, gamma = GAMMA, lam = GAE_LAMBDA):
    """
    By trajectory calculate advantage and reference. Standard step in PPO.
    :param rewards: the tensor of rewards
    :param dones: the tensor of dones
    :param values: the tensor of values
    :param device: the device
    :return: advantage and reference tensors
    """
    advantage = torch.zeros_like(rewards)
    # generalized advantage estimator: smoothed version of the advantage

    # Shift values to align with the next timestep
    values_next = torch.cat([values[1:], torch.zeros_like(values[-1:])], dim=0)
    dones_next = torch.cat([dones[1:], torch.zeros_like(dones[-1:])], dim=0)

    # Compute delta (TD residual)
    delta = rewards + gamma * values_next * (1 - dones_next) - values

    gae = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.size(0))):
        gae = delta[t] + (1 - dones_next[t]) * gamma * lam * gae
        advantage[t] = gae

    reference = advantage + values

    return advantage.to(device), reference.to(device)





def ppo_training_loop():
    """
    the main ppo training loop.
    The function is used as a solitary function with little external dependencies, which means that it defines its only variables and hyperparameters.
    :return: none
    """
    #TODO: when the project gets complicated, can creat a separate caller of the training loop to define the variables during training.
    #note that the parameter section is located in the front part of the function.

    #the getting parameters phase
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    #TODO: put these into hyperparameter section
    max_relator_length = 37
    num_steps = 1000
    num_updates = 20
    env_indices = [0, 1, 2, 3, 4]
    all_presentation = load_all_presentations()
    envs = get_envs(all_presentation, env_indices, max_relator_length)
    nodes_counts = [256, 256]
    minibatch_size = 5000
    agent = PPOAgent(envs, nodes_counts).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=EPS)

    #the update loop, consists rollout and optimisation
    for updates in tqdm(range(num_updates), desc="Training"):

        #get the rollout results
        states, actions, rewards, dones, values, logprobs, explodes = rollout(envs, agent, num_steps, device)

        #get advantage and reference
        advantages, references = get_advantage_and_reference(rewards, dones, values, device)

        #optimising step, the key training loop
        # first reshape things to be more easily accessed
        b_states = states.reshape(
            (-1,) + envs.single_observation_space.shape
        )  # num_envs * num_steps, obs_space.shape
        b_logprobs = logprobs.reshape(-1)  # num_envs * num_steps
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_references = references.reshape(-1)
        b_values = values.reshape(-1)

        batch_size = envs.num_envs * num_steps
        b_inds = np.arange(batch_size)  # indices of batch_size

        for epoch in range(NUM_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]


                _, new_logprob, entropy, new_value = agent.get_action_and_value(b_states[mb_inds], b_actions.long()[mb_inds])  # .long() converts dtype to int64
                logratio = new_logprob - b_logprobs[mb_inds]
                ratio = (logratio.exp())  # pi(a|s) / pi_old(a|s); is a tensor of 1s for epoch=0.
                mb_advantages = b_advantages[mb_inds]

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_value.flatten(), b_references[mb_inds])
                total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy.mean()


                #optimise the network
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    torch.save(agent.state_dict(), "./trained_models/ppo.pt")



"""
this block of the notebook checks the above rollout procedures
"""
if __name__ == "__main__":
    ppo_training_loop()