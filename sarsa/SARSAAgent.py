import numpy as np
import pandas as pd

from generics.AbstractAgent import AbstractAgent


class SARSAAgent(AbstractAgent):

    def __init__(self, train, screen_size):
        super(SARSAAgent, self).__init__(screen_size)
        if train:
            self.q_table = self._init_q_table()
            self.count_table = self._init_count_table()
        self.train = train

        self.gamma = 0.9
        self.alpha = 0.1

        self.state = None
        self.action = None
        self.next_state = None
        self.next_action = None

    def step(self, obs):
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            self.next_state = self._calculate_state(obs)
            self.next_action = self._choose_action(self.next_state)
            if self.train:
                self._update_q_table(obs, self.state, self.action, self.next_state, self.next_action)
            self.state = self.next_state
            self.action = self.next_action
            self.count_table.at[self.state, self.action] += 1
            return self._dir_to_sc2_action(self.action, self._get_unit_pos(self._get_marine(obs)))
        else:
            self.state = self._calculate_state(obs)
            self.action = self._choose_action(self.state)
            return self._SELECT_ARMY

    def _update_q_table(self, obs, state, action, next_state, next_action):
        self._init_q_table_state_if_not_exists(state)
        self._init_q_table_state_if_not_exists(next_state)
        q_value = self.q_table.at[state, action]
        if obs.last() or obs.reward is 1:
            target = obs.reward
        else:
            target = obs.reward + self.gamma * self.q_table.at[next_state, next_action]
        self.q_table.at[state, action] += self.alpha * (target - q_value)

    def _choose_action(self, state):

        def _upper_confidence_bound(state, action):
            self._init_count_table_state_if_not_exists(state)
            # TODO Erklärung dieser Formel einfügen, sehr unübersichtlich
            # Warum dashier funktioniert (.sum().sum()) idk.
            state_visits = self.count_table.loc[[state]].sum().sum()
            return np.sqrt(2 * np.log2(state_visits) / self.count_table.at[state, action])

        self._init_q_table_state_if_not_exists(state)

        expected_bounds = pd.Series(
            [self.q_table.at[state, action] + _upper_confidence_bound(state, action) for action in self._DIRECTIONS],
            index=self._DIRECTIONS.keys(), name="selection_values")
        return expected_bounds.idxmax()

    def _calculate_state(self, obs):
        """
        Shrink the space state by 4
        :param marine_coord:
        :param beacon_coord:
        :return:
        """
        marine = self._get_marine(obs)
        beacon = self._get_beacon(obs)
        marine_coord = self._get_unit_pos(marine)
        beacon_coord = self._get_unit_pos(beacon)
        x_diff, y_diff = np.subtract(marine_coord, beacon_coord)

        if x_diff >= 0:
            x_diff = np.ceil(x_diff / 8)
        else:
            x_diff = np.floor(x_diff / 8)

        if y_diff >= 0:
            y_diff = np.ceil(y_diff / 8)
        else:
            y_diff = np.floor(y_diff / 8)

        return x_diff, y_diff

    def save_model(self, filename):
        print(self.q_table)
        pass

    def load_model(self, filename):
        pass

    def _init_q_table_state_if_not_exists(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.q_table.columns), index=self.q_table.columns, name=state))

    def _init_count_table_state_if_not_exists(self, state):
        if state not in self.count_table.index:
            self.count_table = self.count_table.append(
                pd.Series([1] * len(self.count_table.columns), index=self.count_table.columns, name=state))

    def _init_q_table(self):
        return pd.DataFrame(columns=self._DIRECTIONS.keys(), dtype=np.float64)

    def _init_count_table(self):
        return pd.DataFrame(columns=self._DIRECTIONS.keys(), dtype=np.int64)
