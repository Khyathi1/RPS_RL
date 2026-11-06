import numpy as np

MOVES = ('rock', 'paper', 'scissors')


def reward(agent_action: int, opponent_action: int) -> int:
    """+1 win, 0 tie, -1 loss."""
    if agent_action == opponent_action:
        return 0
    return 1 if (agent_action - opponent_action) % 3 == 1 else -1


class QAgent:
    """Tabular Q-learning with previous opponent move as state."""
    def __init__(self, alpha=0.1, gamma=0.0, epsilon=1.0, epsilon_decay=0.9995,
                 min_epsilon=0.05, seed=42):
        self.q = np.zeros((3, 3), dtype=float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.rng = np.random.default_rng(seed)

    def act(self, state: int, explore=True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(3))
        row = self.q[state]
        choices = np.flatnonzero(np.isclose(row, row.max()))
        return int(self.rng.choice(choices))

    def update(self, state: int, action: int, result: int, next_state: int, terminal=True):
        # Each training episode is a one-round game; no future reward to bootstrap.
        target = result if terminal else result + self.gamma * self.q[next_state].max()
        self.q[state, action] += self.alpha * (target - self.q[state, action])
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, path):
        np.savez(path, q=self.q, alpha=self.alpha, gamma=self.gamma)

    @classmethod
    def load(cls, path, seed=42):
        with np.load(path) as data:
            obj = cls(alpha=float(data['alpha']), gamma=float(data['gamma']), seed=seed)
            obj.q = data['q'].copy()
        return obj
