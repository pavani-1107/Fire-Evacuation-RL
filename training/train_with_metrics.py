import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append('..')

from env.building_env import BuildingEnv
from agents.qlearning import QLearningAgent
from agents.sarsa import SARSAAgent
import matplotlib.pyplot as plt

def train_and_track(agent_type='qlearning', episodes=500):
    """Train agent and track success rate over time."""
    
    env = BuildingEnv(grid_size=(8, 8), fire_spread_prob=0.1)
    
    if agent_type == 'qlearning':
        agent = QLearningAgent(state_size=5000, action_size=5, learning_rate=0.1, discount=0.99)
        print("Training Q-Learning with metrics...")
    else:
        agent = SARSAAgent(state_size=5000, action_size=5, learning_rate=0.1, discount=0.99)
        print("Training SARSA with metrics...")
    
    success_rates = []
    rewards_history = []
    window = 50  # Calculate success rate over last 50 episodes
    
    successes = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        if agent_type == 'sarsa':
            action = agent.choose_action(state)
        
        for step in range(200):
            if agent_type == 'qlearning':
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                agent.learn(state, action, reward, next_state, done)
            else:
                next_state, reward, done, info = env.step(action)
                next_action = agent.choose_action(next_state)
                agent.learn(state, action, reward, next_state, next_action, done)
                action = next_action
            
            total_reward += reward
            state = next_state
            
            if done:
                successes.append(1 if reward == 100 else 0)
                break
        
        if not done:
            successes.append(0)
        
        rewards_history.append(total_reward)
        
        # Calculate rolling success rate
        if len(successes) >= window:
            success_rate = sum(successes[-window:]) / window * 100
        else:
            success_rate = sum(successes) / len(successes) * 100
        
        success_rates.append(success_rate)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} | Success Rate: {success_rate:.1f}%")
    
    return success_rates, rewards_history

# Train both algorithms
q_success, q_rewards = train_and_track('qlearning', episodes=500)
print()
s_success, s_rewards = train_and_track('sarsa', episodes=500)

# Plot success rates
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(q_success, label='Q-Learning', linewidth=2)
plt.plot(s_success, label='SARSA', linewidth=2)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Success Rate (%)', fontsize=12)
plt.title('Success Rate Over Training', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(q_rewards, label='Q-Learning', alpha=0.6)
plt.plot(s_rewards, label='SARSA', alpha=0.6)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Total Reward', fontsize=12)
plt.title('Rewards Over Training', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('success_rate_comparison.png', dpi=150)
print("\nSuccess rate graph saved as success_rate_comparison.png")