"""
All AC moves, and added equivalent to AC moves.
"""

import numpy as np


"""
the follows are common utils when making transformations.
"""

def simplify_relators(state):
    #TODO: check this function works correctly
    """
    funtion to make the relations into most simple form by removing neighbouring inverses
    :param state: presentation to be simplified
    :return simplified state

    This has complexity O(len(state)^2), can be simplified.
    """
    new_state = []

    #first stage, cancel redundant 0
    for i in range(0, len(state)):
        if i == len(state) - 1:
            new_state.append(state[i])
        elif not (state[i] == 0 and state[i+1] == 0):
            new_state.append(state[i])
    state = new_state.copy()
    new_state = []

    #second stage, cancel adjacent inverses
    while True:
        skip = False
        for i in range(0, len(state)):
            if skip:
                skip = False
                continue
            elif i == len(state) - 1:
                new_state.append(state[i])
            elif state[i] != - state[i + 1]:
                new_state.append(state[i])
            else:
                skip = True
                continue
        if len(new_state) == len(state):
            break
        else:
            state = new_state.copy()
            new_state = []

    #if last element is not 0, add a 0 to the end
    if state[-1] != 0:
        state.append(0)

    return state

def get_relators_and_positions(state):
    """
    This function gets the start and end position of each relator.
    :param state:list, the state to be determined
    :return relators: a tuple to 2 lists containing 2 relators, without zero
    :return: positions: list of 2 ints, ending positions of each relator
    """
    state = simplify_relators(state)
    first_position = 0
    for i in range(0, len(state)):
        if state[i] == 0:
            first_position = i
            break
    positions = [first_position, len(state) - 1]
    first_relator = []
    second_relator = []
    for i in range(0, positions[0]):
        first_relator.append(state[i])
    for i in range(positions[0] + 1, positions[1]):
        second_relator.append(state[i])

    return first_relator, second_relator, positions

"""The followings are AC moves and their equivalents"""

def ac1(state, target_relator):
    first_relator, second_relator, positions = get_relators_and_positions(state)
    if target_relator == 1:
        first_relator = first_relator + second_relator
    elif target_relator == 2:
        second_relator = second_relator + first_relator

    new_state = first_relator + [0] + second_relator + [0]
    return simplify_relators(new_state)

def ac2(state, target_relator):
    first_relator, second_relator, positions = get_relators_and_positions(state)
    if target_relator == 1:
        first_relator = [-x for x in first_relator[::-1]]
    if target_relator == 2:
        second_relator = [-x for x in second_relator[::-1]]

    new_state = first_relator + [0] + second_relator + [0]
    return simplify_relators(new_state)

def ac3(state, target_relator, target_generator):
    first_relator, second_relator, positions = get_relators_and_positions(state)
    if target_relator == 1:
        if target_generator == 1:
            first_relator = [1] + first_relator + [-1]
        if target_generator == 2:
            first_relator = [2] + first_relator + [-2]
    if target_relator == 2:
        if target_generator == 1:
            second_relator = [1] + second_relator + [-1]
        if target_generator == 2:
            second_relator = [2] + second_relator + [-2]

    new_state = first_relator + [0] + second_relator + [0]
    return simplify_relators(new_state)


"""
the followings are the calling command and the get reward function
"""

def get_reward(state, new_state, action = None):
    pass

def make_move(state, action):
    """
    function to make the move.
    :param state: list, ACEnv.state param
    :param action: int, action token in args.actions
    :return: new state variable

    action list:
        1: AC1 1 to 12
        2: AC1 2 to 21
        3: AC2 1 to -1
        4: AC2 2 to -2
        5: AC3 1 to 11-1
        6: AC3 1 to 21-2
        7: AC3 2 to 12-1
        8: AC3 2 to 22-2
    """
    state = simplify_relators(state)

    if action == 0:
        state = ac1(state, 1)
    if action == 1:
        state = ac1(state, 2)
    if action == 2:
        state = ac2(state, 1)
    if action == 3:
        state = ac2(state, 2)
    if action == 4:
        state = ac3(state, 1, 1)
    if action == 5:
        state = ac3(state, 1, 2)
    if action == 6:
        state = ac3(state, 2, 1)
    if action == 7:
        state = ac3(state, 2, 2)

    state = simplify_relators(state)

    return state