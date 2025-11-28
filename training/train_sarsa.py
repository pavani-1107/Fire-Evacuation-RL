import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append('..')

from env.building_env import BuildingEnv
from agents.sarsa import SARSAAgent
import matplotlib.pyplot as plt

#Training parameters
episodes = 500
max_steps = 100

#Create enviroment and agent
env = BuildingEnv(grid_size=(5, 5), fire_spread_prob=0.1)
agent = SARSAAgent(state_size=1000, action_size=5, learning_rate=0.1, discount=0.99)

#Track results
episode_rewards = []
success_count = 0

print("Training SARSA Agent...")

for episode in range(episodes):
    state = env.reset()
    action = agent.choose_action(state) #SARSA: choose first action
    total_reward = 0

    for step in range(max_steps):
        #Take action
        next_state, reward, done, info = env.step(action)
        
        #Choose next action
        next_action = agent.choose_action(next_state)

        #Learn (SARSA uses next_action)
        agent.learn(state, action, reward, next_state, next_action, done)

        total_reward += reward
        state = next_state
        action = next_action #SARSA: use the chosen next action

        if done:
            if reward == 100:
                success_count += 1
            break

    episode_rewards.append(total_reward)

    if (episode + 1) % 50 == 0:
        avg_reward = sum(episode_rewards[-50:]) / 50
        success_rate = success_count / (episode + 1) * 100
        print(f"Episode {episode + 1}/{episodes} | Avg Reward: {avg_reward:.2f} | Success Rate: {success_rate:.1f}%")

print(f"Training complete! Final Success Rate: {success_count/episodes*100:.1f}%")

#Plot Results
plt.figure(figsize=(10,5))
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('SARSA Training Progress')
plt.grid(True)
plt.savefig('sarsa_results.png')