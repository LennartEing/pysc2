from absl import app
from generics.env import Env
from generics.runner import Runner
from sarsa.SARSAAgent import SARSAAgent

_CONFIG = dict(
    episodes=1000,
    screen_size=64,
    minimap_size=64,
    visualize=False,
    train=True,
    agent=SARSAAgent,
    load_path='./graphs/200505_1947_train_QLearningAgent',
    save_path='./graphs/'
)


def main(unused_argv):

    agent = _CONFIG['agent'](
        train=_CONFIG['train'],
        screen_size=_CONFIG['screen_size']
    )

    env = Env(
        screen_size=_CONFIG['screen_size'],
        minimap_size=_CONFIG['minimap_size'],
        visualize=_CONFIG['visualize']
    )

    runner = Runner(
        agent=agent,
        env=env,
        train=_CONFIG['train'],
        load_path=_CONFIG['load_path'],
        save_path=_CONFIG['save_path']
    )

    runner.run(episodes=_CONFIG['episodes'])


if __name__ == "__main__":
    app.run(main)