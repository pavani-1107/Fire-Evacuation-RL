import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append('..')

from env.building_env import BuildingEnv
from agents.qlearning import QLearningAgent
import matplotlib.pyplot as plt

# Training parameters
episodes = 500
max_steps = 100

#Create environment and agent
env = BuildingEnv(grid_size=(5, 5), fire_spread_prob=0.1)
agent = QLearningAgent(state_size=1000, action_size=5, learning_rate=0.1, discount=0.99)

#Track results
episode_rewards = []
success_count = 0

print("Training Q-Learning agent...")

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    for step in range(max_steps):
        #Choose and take action
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)

        #Learn
        agent.learn(state, action, reward, next_state, done)

        total_reward += reward
        state = next_state

        if done:
            if reward == 100: #Success
                success_count += 1
            break

    episode_rewards.append(total_reward)

    if(episode + 1) % 50 == 0:
        avg_reward = sum(episode_rewards[-50:]) / 50
        success_rate = success_count / (episode + 1) * 100
        print(f"Episode {episode + 1}/{episode} | Avg Reward: {avg_reward:.2f} | Success Rate: {success_rate:.1f}%")

print(f"Training complete! Final success rate: {success_count/episodes*100:.1f}%")

#Plot Results
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-Learning Training Progress')
plt.grid(True)
plt.savefig('qlearning_results.png')