import numpy as np
import pandas as pd
import sys
import glob

graph = {
  'A' : ['B', 'D'],
  'B' : ['A', 'C'],
  'C' : ['B', 'E'],
  'D' : ['A', 'E', 'F'],
  'E' : ['C', 'D', 'F'],
  'F' : ['D', 'E']
}

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

class td_qlearning:

  alpha = 0.10
  gamma = 0.95

  #initial q function
  q_function = [
      -np.ones(7),      #A
      10 * np.ones(7),  #B
      -np.ones(7),      #C
      -np.ones(7),      #D
      -np.ones(7),      #E
      -np.ones(7)       #F
    ]



  def __init__(self, directory):
    self.trials = data = [[['FE', 'E'], ['ED', 'F'], ['FA', 'E'], ['ED', 'D'], ['DA', 'F'], ['FA', 'N'], ['FD', 'D'], ['DE', 'E'], ['ED', 'F'], ['FA', 'D'], ['DD', '-']], [['EA', 'F'], ['FD', 'N'], ['FD', 'D'], ['DA', 'N'], ['DA', 'E'], ['EB', 'D'], ['DB', 'N'], ['DB', 'N'], ['DB', 'F'], ['FB', 'N'], ['FB', 'D'], ['DB', 'E'], ['EB', 'N'], ['EB', 'F'], ['FB', 'D'], ['DB', 'F'], ['FB', 'D'], ['DB', 'F'], ['FB', 'D'], ['DB', 'F'], ['FB', 'N'], ['FB', 'D'], ['DB', 'N'], ['DB', 'E'], ['EB', 'D'], ['DB', 'A'], ['AB', 'B'], ['BB', '-']], [['FE', 'E'], ['EF', 'D'], ['DE', 'E'], ['EF', 'N'], ['ED', 'N'], ['ED', 'C'], ['CF', 'E'], ['ED', 'C'], ['CD', 'B'], ['BD', '-']], [['EB', 'F'], ['FB', 'N'], ['FB', 'E'], ['EB', 'C'], ['CB', 'B'], ['BB', '-']], [['AB', 'D'], ['DB', 'F'], ['FB', 'E'], ['EB', 'D'], ['DB', 'A'], ['AB', 'D'], ['DB', 'F'], ['FB', 'N'], ['FB', 'E'], ['EB', 'C'], ['CB', 'N'], ['CB', 'E'], ['EB', 'F'], ['FB', 'N'], ['FB', 'N'], ['FB', 'D'], ['DB', 'F'], ['FB', 'D'], ['DB', 'E'], ['EB', 'F'], ['FB', 'N'], ['FB', 'E'], ['EB', 'N'], ['EB', 'C'], ['CB', 'B'], ['BB', '-']], [['ED', 'N'], ['EA', 'D'], ['DD', '-']], [['ED', 'C'], ['CD', 'N'], ['CE', 'E'], ['ED', 'C'], ['CA', 'B'], ['BA', '-']], [['EF', 'D'], ['DF', 'F'], ['FD', 'N'], ['FE', 'E'], ['EC', 'F'], ['FC', 'D'], ['DB', 'N'], ['DB', 'F'], ['FB', 'E'], ['EB', 'D'], ['DB', 'A'], ['AB', 'B'], ['BB', '-']], [['FD', 'D'], ['DF', 'F'], ['FD', 'E'], ['EE', '-']], [['FB', 'N'], ['FB', 'D'], ['DB', 'F'], ['FB', 'N'], ['FB', 'D'], ['DB', 'E'], ['EB', 'N'], ['EB', 'F'], ['FB', 'D'], ['DB', 'F'], ['FB', 'E'], ['EB', 'C'], ['CB', 'B'], ['BB', '-']]]


    self.trials = []
    
    for file in glob.glob(f"{directory}/*.csv"):
      self.trials.append(pd.read_csv(file).values)

  def qvalue(self, state, action):
    # state is a string representation of a state
    # action is a string representation of an action

    stateIndex = stateToIndex[state]
    actionIndex = actionToIndex[action]

    q = self.q_function[stateIndex][actionIndex]

    # Return the q-value for the state-action pair
    return q

  def policy(self, state):
    # state is a string representation of a state

    index = np.argmax(self.q_function[stateToIndex[state]])
    a = indexToAction[index]
    # Return the optimal action under the learned policy
    return a
   
  def temporal_difference(self, state, action):
    self.q_function[stateToIndex[state[0]]][action] += self.alpha * (self.reward(state) + self.gamma * self.q_function[stateToIndex[action]][actionToIndex["N"]] - self.q_function[stateToIndex[state[0]]][action])

  def reward(self, state):
    if state[0] == 'B': 
        return 10
    if state[0] == state[1] and state[0] != 'B' : 
        return -10
    else:
        return -1

