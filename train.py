import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from agent import QAgent, MOVES, reward
from environment import opponent_action


def train(episodes=20000, strategy='repeat70', seed=42, alpha=0.1,
          epsilon_decay=0.9995, min_epsilon=0.05):
    rng = np.random.default_rng(seed)
    agent = QAgent(alpha=alpha, epsilon_decay=epsilon_decay,
                   min_epsilon=min_epsilon, seed=seed + 1)
    previous = int(rng.integers(3))
    records = []
    for i in range(episodes):
        state = previous
        action = agent.act(state)
        opponent = opponent_action(strategy, previous, rng)
        result = reward(action, opponent)
        agent.update(state, action, result, opponent)
        records.append({'episode': i + 1, 'reward': result, 'win': int(result == 1),
                        'draw': int(result == 0), 'loss': int(result == -1),
                        'epsilon': agent.epsilon})
        previous = opponent
    return agent, pd.DataFrame(records)


def evaluate(agent, strategy='repeat70', rounds=5000, seed=2026):
    rng = np.random.default_rng(seed)
    previous = int(rng.integers(3))
    outcomes = []
    for _ in range(rounds):
        action = agent.act(previous, explore=False)
        opponent = opponent_action(strategy, previous, rng)
        outcomes.append(reward(action, opponent))
        previous = opponent
    outcomes = np.array(outcomes)
    return {'win_rate': float(np.mean(outcomes == 1)),
            'draw_rate': float(np.mean(outcomes == 0)),
            'loss_rate': float(np.mean(outcomes == -1)),
            'average_reward': float(outcomes.mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20000)
    parser.add_argument('--strategy', choices=['random', 'repeat70', 'cycle', 'biased_rock'], default='repeat70')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    agent, history = train(args.episodes, args.strategy, args.seed)
    out = Path('outputs')
    out.mkdir(exist_ok=True)
    agent.save(out / 'q_agent.npz')
    history.to_csv(out / 'training_history.csv', index=False)
    print('Q-table (rows=previous opponent move, columns=AI move):')
    print(pd.DataFrame(agent.q, index=MOVES, columns=MOVES).round(3))
    print('Evaluation (exploration OFF):', evaluate(agent, args.strategy))
    print('Saved outputs/q_agent.npz and outputs/training_history.csv')


if __name__ == '__main__':
    main()
