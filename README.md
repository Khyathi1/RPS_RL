# Rock–Paper–Scissors Reinforcement Learning Lab

A beginner-friendly, complete Python project: tabular Q-learning, opponent simulations, evaluation, tests, and an interactive Streamlit dashboard.

## Install

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python train.py --episodes 20000 --strategy repeat70
streamlit run app.py
pytest -q
```

The CLI saves a Q-table to `outputs/q_agent.npz` and training results to `outputs/training_history.csv`. The dashboard trains its own model when you click the Train button.

## How it works

- **State:** opponent's previous move (rock, paper, scissors).
- **Action:** agent's move (rock, paper, scissors).
- **Reward:** win +1, tie 0, loss -1.
- **Algorithm:** epsilon-greedy tabular Q-learning.
- **Episode:** one game round. Since episodes are terminal after one round, the Q update uses immediate reward only (`gamma` has no effect in terminal updates). This is effectively a contextual bandit, implemented using the terminal Q-learning update.
- **Opponent strategies:** `repeat70` (70% repeats previous move, otherwise random), `cycle` (always advances rock → paper → scissors), `biased_rock` (70% rock), and `random` (uniformly random).

The learned agent is evaluated without exploration on fresh opponent rounds. Against a truly random opponent, the theoretical expected win/draw/loss rates are each 1/3; a high win rate is not guaranteed. Against predictable opponents, the agent can exploit patterns.

## Files

- `agent.py`: agent, rewards, Q-table and model save/load.
- `environment.py`: opponent strategies.
- `train.py`: training, evaluation, CLI and CSV output.
- `app.py`: interactive Streamlit dashboard and play mode.
- `tests/test_project.py`: automated tests.

## Experiment ideas

1. Compare evaluation win rates against all four opponent strategies.
2. Compare 1,000 vs 20,000 training episodes.
3. Change epsilon decay and inspect the rolling win-rate graph.
4. Add a state containing the last two opponent moves to capture longer patterns.

## Limitation

The model only observes the last opponent move. A human opponent may change behavior, so the model trained on a simulated opponent may not perform well against a human. The human play mode does not update the Q-table; it uses the trained model as-is.
