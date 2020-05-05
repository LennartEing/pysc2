import pandas as pd
import numpy as np

from generics.AbstractAgent import AbstractAgent

class QLearningAgent(AbstractAgent):

    def __init__(self, train, screen_size):
        super(QLearningAgent, self).__init__(screen_size)
        self.train = train
        self.actions = list(self._DIRECTIONS.keys())
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

        #Save states between iterations
        self.previous_state = None
        self.new_state = None
        self.action_key_taken = None

        #Metavariables
        if self.train:
            self.epsilon = 1
        else:
            self.epsilon = 0
        self.gamma = .9
        self.alpha = .9


    def step(self, obs):
        self.new_state = self._calculate_state(obs)
        action = None
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            if self.train:
                self._learn(self.previous_state, self.action_key_taken, obs.reward, self.new_state)
            self.action_key_taken = self._choose_action(self.new_state)
            action =self._dir_to_sc2_action(self.action_key_taken, self._get_unit_pos(self._get_marine(obs)))
        else:
            action = self._SELECT_ARMY
            self.action_key_taken = 'N'
        self.previous_state = self.new_state
        return action

    def save_model(self, filename):
        if self.epsilon > 0:
            self.epsilon -= 0.2
        self.q_table.to_pickle(filename)

    def load_model(self, filename):
        self.q_table = pd.read_pickle(filename)

    def _learn(self, old_state, action_taken, reward, new_state):
        self._introduce_state_if_unknown(old_state)
        self._introduce_state_if_unknown(new_state)
        prediction = self.q_table.at[old_state, action_taken]
        if new_state is not 'terminal':
            target = reward + self.gamma * self.q_table.loc[new_state].max()
        else:
            target = reward
        self.q_table.at[old_state, action_taken] += self.alpha * (target - prediction)

    def _choose_action(self, state):
        self._introduce_state_if_unknown(state)
        if np.random.uniform() > self.epsilon:
            #Choose from multiple actions.
            #Multiple actions are possible if they have the same expected reward
            possible_actions = self.q_table.loc[state]
            #Shuffle, so that we don't always take the first one, but rather any.
            possible_actions = possible_actions.reindex(np.random.permutation(possible_actions.index))
            action = possible_actions.idxmax()
        else:
            action = np.random.choice(self.actions)
        return action

    def _introduce_state_if_unknown(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

    def _calculate_state(self, obs):
        marine = self._get_marine(obs)
        beacon = self._get_beacon(obs)
        marine_position = self._get_unit_pos(marine)
        beacon_position = self._get_unit_pos(beacon)
        distance_vector = marine_position - beacon_position
        state = ''
        if distance_vector[1] > 0:
            state += 'N'
        elif distance_vector[1] < 0:
            state += 'S'
        if distance_vector[0] > 0:
            state += 'W'
        elif distance_vector[0] < 0:
            state += 'E'
        if state is '':
            state = 'terminal'
        return state
