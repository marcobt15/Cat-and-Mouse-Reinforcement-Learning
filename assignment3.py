import numpy as np
import pandas as pd
import sys
import glob

stateToIndex = {
    'A' : 0,
    'B' : 1,
    'C' : 2,
    'D' : 3,
    'E' : 4,
    'F' : 5,
  }

indexToAction = {
    0 : 'N',
    1 : 'A',
    2 : 'B',
    3 : 'C',
    4 : 'D',
    5 : 'E',
    1 : 'F'
  }

actionToIndex = {
        'N':0,
        'A':1,
        'B':2,
        'C':3,
        'D':4,
        'E':5,
        'F':6
    }

#creating initial q function
q_function = {}
states = ['A', 'B', 'C', 'D', 'E', 'F']
for mouse_state in states:
    for cat_state in states:
        #7 actions
        new_q_list = []
        if mouse_state == 'B':
            new_q_list = 10 * np.ones(7)
        elif mouse_state == cat_state:
            new_q_list = -10 * np.ones(7)
        else:
            new_q_list = -np.ones(7)

        q_function[mouse_state+cat_state] = new_q_list
    

class td_qlearning:

  alpha = 0.10
  gamma = 0.95

  def __init__(self, directory):
    self.trials = [[['FE', 'E'], ['ED', 'F'], ['FA', 'E'], ['ED', 'D'], ['DA', 'F'], ['FA', 'N'], ['FD', 'D'], ['DE', 'E'], ['ED', 'F'], ['FA', 'D'], ['DD', '-']], [['EA', 'F'], ['FD', 'N'], ['FD', 'D'], ['DA', 'N'], ['DA', 'E'], ['EB', 'D'], ['DB', 'N'], ['DB', 'N'], ['DB', 'F'], ['FB', 'N'], ['FB', 'D'], ['DB', 'E'], ['EB', 'N'], ['EB', 'F'], ['FB', 'D'], ['DB', 'F'], ['FB', 'D'], ['DB', 'F'], ['FB', 'D'], ['DB', 'F'], ['FB', 'N'], ['FB', 'D'], ['DB', 'N'], ['DB', 'E'], ['EB', 'D'], ['DB', 'A'], ['AB', 'B'], ['BB', '-']], [['FE', 'E'], ['EF', 'D'], ['DE', 'E'], ['EF', 'N'], ['ED', 'N'], ['ED', 'C'], ['CF', 'E'], ['ED', 'C'], ['CD', 'B'], ['BD', '-']], [['EB', 'F'], ['FB', 'N'], ['FB', 'E'], ['EB', 'C'], ['CB', 'B'], ['BB', '-']], [['AB', 'D'], ['DB', 'F'], ['FB', 'E'], ['EB', 'D'], ['DB', 'A'], ['AB', 'D'], ['DB', 'F'], ['FB', 'N'], ['FB', 'E'], ['EB', 'C'], ['CB', 'N'], ['CB', 'E'], ['EB', 'F'], ['FB', 'N'], ['FB', 'N'], ['FB', 'D'], ['DB', 'F'], ['FB', 'D'], ['DB', 'E'], ['EB', 'F'], ['FB', 'N'], ['FB', 'E'], ['EB', 'N'], ['EB', 'C'], ['CB', 'B'], ['BB', '-']], [['ED', 'N'], ['EA', 'D'], ['DD', '-']], [['ED', 'C'], ['CD', 'N'], ['CE', 'E'], ['ED', 'C'], ['CA', 'B'], ['BA', '-']], [['EF', 'D'], ['DF', 'F'], ['FD', 'N'], ['FE', 'E'], ['EC', 'F'], ['FC', 'D'], ['DB', 'N'], ['DB', 'F'], ['FB', 'E'], ['EB', 'D'], ['DB', 'A'], ['AB', 'B'], ['BB', '-']], [['FD', 'D'], ['DF', 'F'], ['FD', 'E'], ['EE', '-']], [['FB', 'N'], ['FB', 'D'], ['DB', 'F'], ['FB', 'N'], ['FB', 'D'], ['DB', 'E'], ['EB', 'N'], ['EB', 'F'], ['FB', 'D'], ['DB', 'F'], ['FB', 'E'], ['EB', 'C'], ['CB', 'B'], ['BB', '-']]]

    return

    self.trials = []
    
    for file in glob.glob(f"{directory}/*.csv"):
      self.trials.append(pd.read_csv(file).values)

  def qvalue(self, state, action):
    # state is a string representation of a state
    # action is a string representation of an action
    actionIndex = actionToIndex[action]

    q = q_function[state][actionIndex]

    # Return the q-value for the state-action pair
    return q

  def policy(self, state):
    # state is a string representation of a state

    index = np.argmax(q_function[state])
    a = indexToAction[index]
    # Return the optimal action under the learned policy
    return a
   
  def td_q_function_update(self):
    for trial in self.trials:
        for i in range(len(trial)-1):
            state, action = trial[i]
            next_state = trial[i+1][0]

            curr_q_value = q_function[state][actionToIndex[action]]
            reward_next_state = self.reward(next_state)
            best_next_q = max(q_function[next_state])

            q_function[state][actionToIndex[action]] += self.alpha * (reward_next_state + self.gamma * best_next_q - curr_q_value)

  def reward(self, state):
    if state[0] == 'B': 
        return 10
    elif state[0] == state[1] and state[0] != 'B' : 
        return -10
    else:
        return -1

