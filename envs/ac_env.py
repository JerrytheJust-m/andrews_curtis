"""
The Andrews-Curtis environment where each env corresponds to a single chain of transformations.
The environment follows the OpenAI Gym interface.
"""

import numpy as np
from torch.distributed.checkpoint.utils import find_state_dict_object

from envs.ac_moves import make_move, simplify_relators, get_relators_and_positions
import gymnasium as gym


def change_state_length(state, max_relator_length):
    """
    function to make any state into states of required format, required format see README
    :param state: (list of ints), state to change
    :param max_relator_length: (int8) max length of a single relator
    :return: new_state: (list of ints), new state
    :return: explode: (Bool), whether relator length exceeds max relator length
    """
    new_state = simplify_relators(state)
    first_relator, second_relator, positions = get_relators_and_positions(new_state)
    #explode if relator length larger than allowed length, return original state
    if len(first_relator) >= max_relator_length or len(second_relator) >= max_relator_length:
        explode = True
        new_state = state.copy()
    else:
        explode = False
        first_relator += [0]*(max_relator_length - len(first_relator))
        second_relator += [0]*(max_relator_length - len(second_relator))
        new_state = first_relator + second_relator

    return new_state, explode



class ACEnv(gym.Env):
    """
    The environment
    """
    def __init__(self, initial_state, max_relator_length):
        """
        :param initial_state: list, the initial presentation
        :param max_relator_length: int, max length of a single relator
        """
        super(ACEnv, self).__init__()

        #normalisation of initial_state to required length
        self.max_relator_length = max_relator_length
        self.initial_state, self.initial_explode = change_state_length(initial_state, self.max_relator_length)
        assert self.initial_explode == False, "the max relator length of initial presentations exceeds bounds"
        self.explode = self.initial_explode #explode value changes only the trajectory, while initial_explode doesn't

        #action space consists of discrete number labels of the actions
        #TODO: when actions are enriched, this argument should change accordingly
        self.action_space = gym.spaces.Discrete(8)

        #the observation space. length of a state is fixed by the initial state to be suitable for nn input
        low = np.ones(len(self.initial_state), dtype=np.int8) * (-2)
        high = np.ones(len(self.initial_state), dtype=np.int8) * 2
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.int8)

        #parameters related to state.
        self.state = self.initial_state.copy() #self.state this the current state, subject to change
        #Note that the format of a state is introduced in detail in the README file.
        self.steps = 0 #track of number of steps

        self.state_memory = [simplify_relators(initial_state)] #this list remembers selectively the state history
                            #Note that memorizes simplified form
                            #TODO: memory criteria to be determined
        self.move_history = [] #this list remembers EVERY move in this env
        self.done = len(self.state) == 4 or self.explode #whether reduced to trivial presentation

    def reset(self, *, seed=None, return_info=False, options=None):
        """Resets the environment to its initial state, in Gym interface."""
        self.state = self.initial_state.copy()
        self.steps = 0
        self.done = len(self.state) == 4 or self.initial_explode
        self.state_memory = [simplify_relators(self.initial_state)]
        self.move_history = []
        return np.array(self.state, dtype=np.int8), {}

    def step(self, action):
        """
        make a new step, and do the recordings
        :param self.state
        :param action
        :return: new state, state memory, history of moves
        """
        #TODO: before renewing state, remember it using memory criteria
        self.steps += 1
        old_state = self.state.copy()
        self.state = make_move(self.state, action)
        self.state, self.explode = change_state_length(self.state, self.max_relator_length)
        self.move_history.append(action)
        self.done = len(self.state) == 4 or self.explode

        reward = 100 * self.done * (1 - self.explode) - sum(1 for x in self.state if x != 0) * (1 - self.done*(1 - self.explode))

        return np.array(self.state, dtype=np.int8), reward, self.done, self.explode, {}

    """
    All basic env properties are now complete. The below are only additional functions.
    """
