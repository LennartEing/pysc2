"""
Koordination des schrittweisen Ablaufs und
regelmäßiger Tasks (z.B. Speichern der Ergebnisse)

"""


import datetime
import os
import tensorflow as tf


class Runner:

    def __init__(self, agent, env, train, load_path, save_path):

        self.agent = agent
        self.env = env
        self.train = train  # run only or train_model model?

        self.score = 0  # store for the scores of an episode
        self.episode = 1  # episode counter

        self.save_path = save_path + datetime.datetime.now().strftime("%y%m%d_%H%M") \
                         + ('_train_' if self.train else 'run_') \
                         + type(agent).__name__

        print(os.path.isfile(load_path))
        if not self.train and load_path is not None and os.path.isfile(load_path):
            self.agent.load_model(load_path)

    def summarize(self):
        if self.train and self.episode % 10 == 0:
            self.agent.save_model(self.save_path)
            try:
                self.agent.update_target_model()
            except AttributeError:
                ...
        self.episode += 1
        self.score = 0

    def run(self, episodes):
        while self.episode <= episodes:
            obs = self.env.reset()
            while True:
                action = self.agent.step(obs)
                if obs.last():
                    break
                obs = self.env.step(action)
                self.score += obs.reward
            self.summarize()
