import sys
sys.path.append('..')

from env.building_env import BuildingEnv
from agents.qlearning import QLearningAgent

#Test different hyperparameters
learning_rates = [0.1, 0.3, 0.5]
discount_factors = [0.9, 0.95, 0.99]

episodes = 300
max_steps = 100

print("Hyperparameter Tuning for Q-Learning Agent")

results = []

for lr in learning_rates:
    for gamma in discount_factors:
        print(f"\nTesting: Learning Rate = {lr}, Discount = {gamma}")

        env = BuildingEnv(grid_size=(5,5), fire_spread_prob=0.1)
        agent = QLearningAgent(state_size=1000, action_size=5, learning_rate=lr, discount=gamma)
        success_count = 0
        total_rewards = []

        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                action = agent.choose_action(state)
                next_state, reward,done, info = env.step(action)
                agent.learn(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state

                if done:
                    if reward == 100:
                        success_count += 1
                    break

            total_rewards.append(episode_reward)

        success_rate = success_count / episodes * 100
        avg_reward = sum(total_rewards) / episodes

        results.append({
            'lr': lr,
            'gamma': gamma,
            'success_rate': success_rate,
            "avg_reward": avg_reward
        })

        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Avg Reward: {avg_reward:.2f}%")

#Print Summary Table
print("SUMMARY TABLE")
print(f"{'LR':<8} {'GAMMA':<8} {'Success %':<12} {'Avg Reward':<12}")

for r in results:
    print(f"{r['lr']:<8} {r['gamma']:<8} {r['success_rate']:<12.1f} {r['avg_reward']:<12.2f}")

#Find Best
best = max(results, key=lambda x: x['success_rate'])
print(f"Best: LR={best['lr']}, Gamma={best['gamma']} → {best['success_rate']:.1f}% success")
