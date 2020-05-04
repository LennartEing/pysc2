import pandas as pd
import numpy as np

class QLearningTable:

    def __init__(self, actions, learning_rate=0.9, discount_factor= 0.9, epsilon_function=0.5):
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_function = epsilon_function
        self.q_table = pd.DataFrame(columns=actions, dtype=np.float64)

    def choose_action(self, obs):
        self._introduce_state_if_not_exists(obs)
        if np.random.uniform() > self.epsilon_function():
            #Choose action with highest reward, if there are more than 1 reindex and choose randomly.
            possible_actions = self.q_table.loc[[obs]]
            possible_actions = possible_actions.reindex(np.random.permutation(possible_actions.index))
            action = possible_actions.idxmax
        else:
            action = np.random.choice(self.actions)
        return action


    def learn(self, initial_state, action_taken, reward_function, new_state):
        self._introduce_state_if_not_exists(new_state)
        self._introduce_state_if_not_exists(initial_state)
        #Predict the reward that was expected when taking the action
        prediction = self.q_table.loc[initial_state, action_taken]
        #Needed for learning_processes with endstates.
        if new_state != 'terminal':
            target = reward_function(initial_state, action_taken, new_state) + self.discount_factor * self.q_table.loc[new_state].max()
        else:
            target = reward_function(initial_state, action_taken, new_state)
        #update the q_table
        self.q_table.loc[initial_state, action_taken] += self.learning_rate * (target - prediction)

    def _introduce_state_if_not_exists(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))