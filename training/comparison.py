import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append('..')

from env.building_env import BuildingEnv
from agents.qlearning import QLearningAgent
from agents.sarsa import SARSAAgent
import matplotlib.pyplot as plt

#Training parameters
episodes = 500
max_steps = 100
num_runs = 3 # Run multiple times and average

print('Comparing Q-Learning vs SARSA')

#Store results
qlearning_all_rewards = []
sarsa_all_rewards = []
qlearning_success_rates = []
sarsa_success_rates = []

for run in range(num_runs):
    print(f"Run {run + 1}/{num_runs}")

    #Train Q-Learning
    env = BuildingEnv(grid_size=(5, 5), fire_spread_prob=0.1)
    q_agent = QLearningAgent(state_size=1000, action_size=5)
    q_rewards = []
    q_success = 0

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = q_agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            q_agent.learn(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                if reward == 100:
                    q_success += 1
                break

        q_rewards.append(total_reward)

    qlearning_all_rewards.append(q_rewards)
    qlearning_success_rates.append(q_success / episodes * 100)

    #Train SARSA
    env = BuildingEnv(grid_size=(5, 5), fire_spread_prob=0.1)
    s_agent = SARSAAgent(state_size=1000, action_size=5)
    s_rewards = []
    s_success = 0

    for episode in range(episodes):
        state = env.reset()
        action = s_agent.choose_action(state)
        total_reward = 0

        for step in range(max_steps):
            next_state, reward, done, info = env.step(action)
            next_action = s_agent.choose_action(next_state)
            s_agent.learn(state, action, reward, next_state, next_action, done)
            total_reward += reward
            state = next_state
            action = next_action
            if done:
                if reward == 100:
                    s_success += 1
                break
        s_rewards.append(total_reward)

    sarsa_all_rewards.append(s_rewards)
    sarsa_success_rates.append(s_success/episodes * 100)

#Calculate averages
import numpy as np
q_avg = np.mean(qlearning_all_rewards, axis = 0)
s_avg = np.mean(sarsa_all_rewards, axis = 0)

print(f"\nResults:")
print(f"Q-Learning - Avg Success Rate: {np.mean(qlearning_success_rates):.1f}%")
print(f"SARSA - Avg Success Rate: {np.mean(sarsa_success_rates):.1f}%")

#Plot Comparison
plt.figure(figsize=(12,5))
plt.plot(q_avg, label = 'Q-Learning', alpha=0.8)
plt.plot(s_avg, label = 'SARSA', alpha=0.8)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Q-Learning vs SARSA')
plt.legend()
plt.grid(True)
plt.savefig('comparison.png')