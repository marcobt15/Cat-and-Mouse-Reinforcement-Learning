import numpy as np
import pandas as pd
import sys
import glob #in python standard library

indexToAction = {
  0 : 'N',
  1 : 'A',
  2 : 'B',
  3 : 'C',
  4 : 'D',
  5 : 'E',
  6 : 'F'
}

actionToIndex = {
  'N' : 0,
  'A' : 1,
  'B' : 2,
  'C' : 3,
  'D' : 4,
  'E' : 5,
  'F' : 6
}    

class td_qlearning:

  alpha = 0.10
  gamma = 0.95

  def __init__(self, directory):

    #creating initial q function
    self.q_function = {}
    states = ['A', 'B', 'C', 'D', 'E', 'F']
    for mouse_state in states:
        for cat_state in states:
            state = mouse_state+cat_state
            curr_q_value = self.reward(state)
            self.q_function[state] = [curr_q_value for _ in range(7)]

    self.trials = []
    
    for file in glob.glob(f"{directory}/*.csv"):
      self.trials.append(pd.read_csv(file, header=None).values)

    self.train()

  def reward(self, state):
    if state[0] == 'B': 
        return 10
    elif state[0] == state[1] and state[0] != 'B' : 
        return -10
    else:
        return -1

  def train(self):
    for _ in range(20):
      for trial in self.trials:
        for i in range(len(trial)-1):
          state, action = trial[i]
          next_state = trial[i+1][0]

          curr_q_value = self.q_function[state][actionToIndex[action]]
          reward_next_state = self.reward(next_state)
          best_next_q = max(self.q_function[next_state])

          new_q = curr_q_value + self.alpha * (reward_next_state + self.gamma * best_next_q - curr_q_value)

          self.q_function[state][actionToIndex[action]] = new_q

  def qvalue(self, state, action):
    # state is a string representation of a state
    # action is a string representation of an action
    actionIndex = actionToIndex[action]
    q = self.q_function[state][actionIndex]

    # Return the q-value for the state-action pair
    return q

  def policy(self, state):
    # state is a string representation of a state
    index = np.argmax(self.q_function[state])
    a = indexToAction[index]
    # Return the optimal action under the learned policy
    return a
  
  def printNicely(self):
    for key, value in self.q_function.items():
      print(key + ": ")
      actions = ['N', 'A', 'B', 'C', 'D', 'E', 'F']
      for i, val in enumerate(value):
        print(actions[i] + ": " + str(val), end=" ")
      print()
      print()

test = td_qlearning("Examples\\Example0\\Trials")

print(test.printNicely())

#For example 0
print(test.policy('EB'))
print(test.qvalue('DA','E'))

#For example 1
# print(test.policy('FC'))#D
# print(test.policy('FC'))#D
# print(test.policy('AC'))#NBD
# print(test.policy('FC'))#D
# print(test.policy('FB'))#DE
# print(test.policy('AA'))#NBD
# print(test.policy('AA'))#NBD
# print(test.policy('EF'))#N

# print(test.qvalue('FC','N'))#-1
# print(test.qvalue('FC','E'))#-1
# print(test.qvalue('AC','D'))#-1
# print(test.qvalue('FC','E'))#-1
# print(test.qvalue('FB','D'))#5.721249999999985
# print(test.qvalue('AA','N'))#-10
# print(test.qvalue('AA','D'))#-10
# print(test.qvalue('EF','N'))#5.000492195194237

#for example 2
# print(test.policy('DC'))#A
# print(test.policy('AE'))#B
# print(test.policy('DA'))#NAEF
# print(test.policy('EC'))#C

# print(test.qvalue('DC','E'))#5.721249999999684
# print(test.qvalue('AE','B'))#8.499999999999993
# print(test.qvalue('DA','E'))#-1
# print(test.qvalue('EC','N'))#-10.49999999999982