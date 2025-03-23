"""
This file is the rollout phase of all training methods. Inputs and outputs are similar with ptan,
except only step_size = 1 is used, and only supports gym vectorised environment.
Note that for early terminating environments, use "ignore done environment" approach and reset the
environment with same initial condition, this is automatically included in gymnasium.
"""

from tqdm import tqdm
import torch

import sys



def rollout(envs, agent, num_steps, device):
    """
    the rollout phrase given which environments to explore.
    :param envs: vectorised environments, containing num_envs many different independent environments.
    :param agent: the A2C agent used in PPO.
    :param num_steps: the max number of steps taken in the rollout. np.int8
    :param device: the device on which the rollout is executed.
    :return: states, actions, rewards, dones, values, logprobs, explodes
    """
    #initialize things
    states = torch.zeros((num_steps, envs.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, envs.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((num_steps, envs.num_envs)).to(device)
    dones = torch.zeros((num_steps, envs.num_envs)).to(device)
    explodes = torch.zeros((num_steps, envs.num_envs)).to(device)
    values = torch.zeros((num_steps, envs.num_envs)).to(device) #values of the states
    logprobs = torch.zeros((num_steps, envs.num_envs)).to(device) #the logprob of the actions in the trajectory


    #record initial information
    next_state = torch.tensor(envs.reset()[0], dtype=torch.float32).to(device)
    next_done = torch.zeros(envs.num_envs).to(device)
    next_explode = torch.zeros(envs.num_envs).to(device)

    #beginning rollout loop
    for step in range(0, num_steps):

        for i in range(envs.num_envs):
            if next_done[i] and not next_explode[i]:
                print("wow !")
        #record and get next move observations
        states[step] = next_state
        dones[step] = next_done
        explodes[step] = next_explode
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_state)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        #take next step in envs and record the reward
        next_state, reward, next_done, next_explode, _ = envs.step(action.cpu().numpy())
        rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
        next_done = torch.tensor(next_done, dtype=torch.float32).to(device)
        next_explode = torch.tensor(next_explode, dtype=torch.float32).to(device)


    return states, actions, rewards, dones, values, logprobs, explodes