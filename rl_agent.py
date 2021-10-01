# !/usr/bin/env python3
# File:     10_rl: rl_agent.py
# Author:   Lukas Malek
# Date:     12.04.2021
# Course:   KUI 2021
#
# SUMMARY OF THE FILE
# Finding the right policy for each state using reinforcement learning
# Undefined maze where the rewards and probabilities are unknown.
# Using Q-Values algorithm with exploration and epsilon-greedy function
#
# SOURCE
# Main source of inspiration for this assignemnt was from these materials
# https://cw.fel.cvut.cz/wiki/_media/courses/b3b33kui/prednasky/08_rl.pdf
# https://cw.fel.cvut.cz/wiki/_media/courses/b3b33kui/prednasky/09_rl.pdf
# https://inst.eecs.berkeley.edu/~cs188/fa18/assets/slides/lec11/FA18_cs188_lecture11_reinforcement_learning_II_1pp.pdf
# Generally speaking, I was inspired by the algorithm from ctu lectures, but
# used the Berkeley equations which are have found more straighforward and
# easier to understand
#
# CREATOR
# A sandbox for playing with the HardMaze
# @author: Tomas Svoboda
# @contact: svobodat@fel.cvut.cz
# @copyright: (c) 2017, 2018

import kuimaze
import numpy as np
import sys
import os
import gym
import random
import copy
import time

MAP = 'maps/normal/normal11.bmp'
# MAP = 'maps_difficult/maze50x50_empty01.png'
# MAP = 'maps/easy/easy4.bmp'
# MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), MAP)

################################## CUSTOM MAP ##################################
REWARD_NORMAL_STATE = -0.04
REWARD_GOAL_STATE = 1
REWARD_DANGEROUS_STATE = -1
GRID_WORLD3 = [[[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0]],
               [[255, 255, 255], [0, 0, 0], [255, 0, 0], [0, 255, 0]],
               [[0, 0, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]]
GRID_WORLD3_REWARDS = [[REWARD_NORMAL_STATE, REWARD_NORMAL_STATE, REWARD_NORMAL_STATE, REWARD_GOAL_STATE],
                       [REWARD_NORMAL_STATE, 0, REWARD_GOAL_STATE /
                           2, REWARD_DANGEROUS_STATE],
                       [REWARD_NORMAL_STATE, REWARD_NORMAL_STATE, REWARD_NORMAL_STATE, REWARD_NORMAL_STATE]]

############################### GLOBAL VARIABLES ###############################

PROBS = [1, 0, 0, 0]
GRAD = (0, 0)
SKIP = False
VERBOSITY = 0  # 0=no GUI, 1=only the final policy, 2=all iterations
TIME_LIMIT = False  # if False we use q-value convergation, otherwise time limit


def wait_n_or_s():
    '''
    Stop the execution until the key 'n' or 's' pressed.
    Used during evaluation to pause and ponder
    press n - next, s - skip to end ... write into terminal
    '''
    def wait_key():
        result = None
        if os.name == 'nt':
            import msvcrt
            # https://cw.felk.cvut.cz/forum/thread-3766-post-14959.html#pid14959
            result = chr(msvcrt.getch()[0])
        else:
            import termios
            fd = sys.stdin.fileno()

            oldterm = termios.tcgetattr(fd)
            newattr = termios.tcgetattr(fd)
            newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSANOW, newattr)
            try:
                result = sys.stdin.read(1)
            except IOError:
                pass
            finally:
                termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
        return result
    global SKIP
    x = SKIP
    while not x:
        key = wait_key()
        x = key == 'n'
        if key == 's':
            SKIP = True
            break


def get_visualisation(table):
    '''
    Visualise the maze, current state is green, the states we visited in
    current epizode are with light green, the states with negative rewards are
    red, the start position is dark blue, and thestate with positive reward
    is dark blue 
    '''
    ret = []
    for i in range(len(table[0])):
        for j in range(len(table)):
            ret.append({'x': j, 'y': i, 'value': [
                       table[j][i][0], table[j][i][1],
                       table[j][i][2], table[j][i][3]]})
    return ret


def exploration_function(next_state, k, number_of_visits, q_table):
    '''
    Calculate the maximum value from all possible actions from the Q-value
    of next_state and action a' + exploitation coefficient divided by the number
    of times we visited the state S. More about the formula on slide 13 of
    https://inst.eecs.berkeley.edu/~cs188/fa18/assets/slides/lec11/FA18_cs188_lecture11_reinforcement_learning_II_1pp.pdf
    :param next_state: list of coordinates of the next state
    :param k: coefficient how much to explore (more info in the berkeley lectures)
    :param number of visits: list with every state and how many times was visited 
    :param q_table: list of q_table values for each state
    :return the values of the best q_value from all possible actions
    '''
    q_values = list()
    for action in [0, 1, 2, 3]:
        q_values.append(q_table[next_state[0]][next_state[1]][action] +
                        k/number_of_visits[next_state[0]][next_state[1]])
    return max(q_values)


def epsilon_greedy_exploration(state, q_table, iteration):
    '''
    Return the best policy for the current state. Find the highest
    Q_value from all directions and return it's action.
    Using epsilon-greedy exploration which sometimes choose the random
    action to explore the enviroment
    :param state: list of coordinates of the current state
    :param q_table: list of q_table values for each state
    :param iteration: which iteration it is (we decrease randomness it in time)
    :return the optimal action to take with the change to being random
    '''
    q_values = list()
    if random.randint(1, 40) == 1:  # probability 2.5% to go random
        return random.randint(0, 3)
    return policy_evaluation(state, q_table)


def policy_evaluation(state, q_table):
    '''
    Return the best policy for the current state. Find the highest
    Q_value from all directions and return it's action
    :param state: list of coordinates of the current state
    :param q_table: list of q_table values for each state
    :return the optimal action to take with the change to being random
    '''
    q_values = list()
    for action in [0, 1, 2, 3]:
        q_values.append(q_table[state[0]][state[1]][action])
    return q_values.index(max(q_values))


def return_policy(env, q_table):
    '''
    Return the best policy for every state from env.get_all_states. It finds 
    the highest Q_value from all directions and return it's action. Used for
    the BRUTE evaluation system, which requires policy for each state
    :param env: the maze enviroment
    :param q_table: list of q_table values for each state
    :return the set{coord:action} of optimal policy for each state
    '''
    policy = dict()
    for state in env.get_all_states():
        policy[state.x, state.y] = policy_evaluation(state, q_table)
    return policy


def q_learning(used_env, maze_size, q_table):
    '''
    I used the algorithm on slide 19 from the following lecture
    https://cw.fel.cvut.cz/wiki/_media/courses/b3b33kui/prednasky/09_rl.pdf
    The equations are replaced by the ones from slide 6 and 13 from bercley lecture
    https://inst.eecs.berkeley.edu/~cs188/fa18/assets/slides/lec11/FA18_cs188_lecture11_reinforcement_learning_II_1pp.pdf
    :param used_env: the maze enviroment
    :param q_table: list of q_table values for each state
    :return the set{coord:action} of optimal policy for each state
    '''
    ########################### VALUE INITIALIZATION ###########################
    k = 0.15        # coefficient how much to explore 0.12
    discount = 0.9  # discount factor gamma
    alpha = 0.15    # learning rate
    number_of_visits = np.ones([maze_size[0], maze_size[1]], dtype=int)
    path = list()
    last_path = list()
    # used_env.action_space.np_random.seed(123)   # not random
    used_env.action_space.np_random.seed()        # random
    obv = used_env.reset()
    state = obv[0:2]
    is_done = False
    iteration = 0  # number of iteration
    start_time = time.time()

    ########################## REINFORCEMENT LEARNING ##########################
    while time.time() - start_time < 19.5:  # time limit 20s in BRUTE
        iteration += 1
        action = epsilon_greedy_exploration(state, q_table, iteration)
        path.append(action)
        obv, reward, is_done, _ = used_env.step(action)
        next_state = obv[0:2]

        ######################### Q_LEARNING EQUATION ##########################
        q_value = q_table[state[0]][state[1]][action]
        # if deterministic and hits the wall assign -inf to to q-value
        # TODO: IMPORTANT
        # turned off only for BRUTE, as it is having some troubles with it
        # uncomment lines 219-221 and ident the lines 222-223
        # if next_state == state and PROBS == [1, 0, 0, 0]:
        #     q_table[state[0]][state[1]][action] = float('-inf')
        # else:
        q_table[state[0]][state[1]][action] = (1 - alpha)*q_value + alpha*(
            reward + discount * (exploration_function(next_state, k, number_of_visits, q_table)))
        number_of_visits[state[0], state[1]] += 1
        state = next_state

        #################### IN TERMINAL STATE START AGAIN #####################
        if is_done:
            obv = used_env.reset()
            state = obv[0:2]
            if path == last_path and TIME_LIMIT == False:  # if true optimal policy is found
                break
            last_path = copy.deepcopy(path)
            path = list()

        ############################ VISUALISATION #############################
        if VERBOSITY > 1:
            used_env.visualise(get_visualisation(q_table))
            used_env.render()
            wait_n_or_s()
    print('Number of iterations:', iteration)
    policy = return_policy(used_env, q_table)
    return policy


def learn_policy(used_env):
    '''
    This function is called by the BRUTE evaluation, return the policy
    as dictionary, the policy is the return from q_learning function, which
    use the Q-learning algortihm to return the best policy how to go
    from the start state to terminate state with the highest reward.
    :param used_env: the maze enviroment
    :return the set{coord:action} of optimal policy for each state
    '''
    x_dims = used_env.observation_space.spaces[0].n
    y_dims = used_env.observation_space.spaces[1].n
    maze_size = tuple((x_dims, y_dims))
    # number of possible actions from state (for us UP,DOWN,LEFT,RIGHT)
    num_actions = used_env.action_space.n
    q_table = np.zeros([maze_size[0], maze_size[1], num_actions], dtype=float)
    policy = q_learning(used_env, maze_size, q_table)

    ############################## VISUALISATION ###############################
    if VERBOSITY > 0:
        obv = used_env.reset()
        state = obv[0:2]
        is_done = False
        while not is_done:
            action = policy_evaluation(state, q_table)
            obv, reward, is_done, _ = used_env.step(action)
            next_state = obv[0:2]
            used_env.visualise(get_visualisation(q_table))
            used_env.render()
            wait_n_or_s()
            state = next_state
    return policy


if __name__ == "__main__":
    '''
    Initialize the maze and call the learn policy function. This
    __main__ can differ from the version which is used in BRUTE
    '''
    # env = kuimaze.HardMaze(map_image=GRID_WORLD3, probs=PROBS,
    #                        grad=GRAD, node_rewards=GRID_WORLD3_REWARDS)
    env = kuimaze.HardMaze(map_image=MAP, probs=PROBS,
                           grad=GRAD)
    if VERBOSITY > 0:
        print('====================')
        print('works only in terminal! NOT in IDE!')
        print('press n - next')
        print('press s - skip to end')
        print('====================')
    learn_policy(env)
