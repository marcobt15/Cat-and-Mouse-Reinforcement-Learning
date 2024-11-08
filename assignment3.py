import pandas as pd
import glob #in python standard library

class td_qlearning:

  alpha = 0.10
  gamma = 0.95

  def __init__(self, directory):
    #representation of the graph
    self.graph = {
        'A': ['B', 'D'], 
        'B': ['A', 'C'], 
        'C': ['B', 'E'], 
        'D': ['A', 'E', 'F'], 
        'E': ['C', 'D', 'F'], 
        'F': ['D', 'E']
    }

    #creating initial q function
    self.q_function = {}
    states = ['A', 'B', 'C', 'D', 'E', 'F']
    actions = ['N', 'A', 'B', 'C', 'D', 'E', 'F']
    for mouse_state in states:
        for cat_state in states:
            state = mouse_state+cat_state
            curr_q_value = self.reward(state)

            #Only adding actions that are valid
            current_state_q_values = {}
            for action in actions:
              if action == 'N' or action in self.graph[mouse_state]:
                current_state_q_values[action] = curr_q_value
            
            self.q_function[state] = current_state_q_values

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
    for _ in range(1000): #1000 was chosen because it leads to convergence and doesn't take too long
      for trial in self.trials:
        for i in range(len(trial)-1):
          state, action = trial[i]
          next_state = trial[i+1][0]

          curr_q_value = self.q_function[state][action]
          current_reward_state = self.reward(state)
          best_next_q = max(self.q_function[next_state].values())

          new_q = curr_q_value + self.alpha * (current_reward_state + self.gamma * best_next_q - curr_q_value)

          self.q_function[state][action] = new_q

  def qvalue(self, state, action):
    # state is a string representation of a state
    # action is a string representation of an action

    # Return the q-value for the state-action pair
    return self.q_function[state][action]

  def policy(self, state):
    # state is a string representation of a state

    # Return the optimal action under the learned policy
    possible_actions = self.q_function[state]
    best_action = None
    max_q_value = float('-inf')
    for key, value in possible_actions.items():
       if value > max_q_value:
          best_action = key
          max_q_value = value

    return best_action