import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from agent import MOVES, reward
from train import train, evaluate

st.set_page_config(page_title='RPS RL Lab', page_icon='🎮', layout='wide')
st.title('🎮 Rock–Paper–Scissors RL Lab')
st.caption('Train a tabular Q-learning agent against predictable and random opponents.')

with st.sidebar:
    st.header('Training settings')
    strategy = st.selectbox('Opponent strategy', ['repeat70', 'cycle', 'biased_rock', 'random'])
    episodes = st.slider('Training rounds', 1000, 50000, 20000, step=1000)
    seed = st.number_input('Random seed', min_value=0, value=42, step=1)
    if st.button('Train / retrain agent', type='primary'):
        agent, history = train(episodes=episodes, strategy=strategy, seed=int(seed))
        st.session_state.agent = agent
        st.session_state.history = history
        st.session_state.strategy = strategy
        st.session_state.last_human_move = None
        st.session_state.score = {'AI': 0, 'You': 0, 'Draw': 0}
        st.session_state.message = 'Training complete!'

if 'agent' not in st.session_state:
    st.info('Choose training settings and click **Train / retrain agent** in the sidebar.')
    st.stop()

agent = st.session_state.agent
history = st.session_state.history
metrics = evaluate(agent, st.session_state.strategy)
a, b, c, d = st.columns(4)
a.metric('AI win rate', f"{metrics['win_rate']:.1%}")
b.metric('Draw rate', f"{metrics['draw_rate']:.1%}")
c.metric('Loss rate', f"{metrics['loss_rate']:.1%}")
d.metric('Average reward', f"{metrics['average_reward']:+.3f}")

st.subheader('Training performance')
window = max(50, min(500, len(history) // 20))
plot_data = history[['episode', 'win', 'draw', 'loss']].copy()
for col in ['win', 'draw', 'loss']:
    plot_data[col] = plot_data[col].rolling(window, min_periods=1).mean()
long_data = plot_data.melt('episode', var_name='Outcome', value_name='Rolling rate')
st.plotly_chart(px.line(long_data, x='episode', y='Rolling rate', color='Outcome',
                        title=f'Rolling outcome rates ({window} rounds)'), use_container_width=True)

st.subheader('Learned Q-table')
q_df = pd.DataFrame(agent.q, index=[f'Previous: {m}' for m in MOVES],
                    columns=[f'AI: {m}' for m in MOVES])
st.dataframe(q_df.style.format('{:.3f}'), use_container_width=True)
st.caption('A larger Q-value means a better estimated immediate reward in that state.')

st.subheader('Play against the trained AI')
st.write('The AI uses your previous move to predict its next best action. The first move is random.')
if 'score' not in st.session_state:
    st.session_state.score = {'AI': 0, 'You': 0, 'Draw': 0}
if 'last_human_move' not in st.session_state:
    st.session_state.last_human_move = None
if 'message' not in st.session_state:
    st.session_state.message = ''

cols = st.columns(3)
for i, move in enumerate(MOVES):
    if cols[i].button(move.title(), use_container_width=True):
        previous = st.session_state.last_human_move
        ai = int(agent.rng.integers(3)) if previous is None else agent.act(previous, explore=False)
        result = reward(ai, i)
        winner = 'AI' if result == 1 else 'You' if result == -1 else 'Draw'
        st.session_state.score[winner] += 1
        st.session_state.last_human_move = i
        st.session_state.message = f'You: {move.title()} | AI: {MOVES[ai].title()} | Result: {winner}'

st.write(st.session_state.message)
st.write('Score:', st.session_state.score)
if st.button('Reset human match'):
    st.session_state.last_human_move = None
    st.session_state.score = {'AI': 0, 'You': 0, 'Draw': 0}
    st.session_state.message = ''
    st.rerun()

st.download_button('Download training history (CSV)', history.to_csv(index=False),
                   file_name='training_history.csv', mime='text/csv')
