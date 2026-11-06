import numpy as np
from agent import QAgent, reward
from environment import opponent_action
from train import train, evaluate


def test_rewards():
    for move in range(3):
        assert reward(move, move) == 0
        assert reward((move + 1) % 3, move) == 1
        assert reward((move + 2) % 3, move) == -1


def test_q_update():
    agent = QAgent(alpha=0.5)
    agent.update(0, 1, 1, 0)
    assert agent.q[0, 1] == 0.5


def test_cycle():
    assert opponent_action('cycle', 0, np.random.default_rng(1)) == 1


def test_training_learns_repeat():
    agent, history = train(episodes=20000, strategy='repeat70')
    assert len(history) == 20000
    assert evaluate(agent, 'repeat70')['win_rate'] > 0.6


def test_random_not_exploitable():
    agent, _ = train(episodes=10000, strategy='random')
    assert evaluate(agent, 'random', rounds=10000)['win_rate'] < 0.38
