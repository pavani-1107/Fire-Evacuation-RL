import sys
sys.path.append('..')

from env.building_env import BuildingEnv
from agents.sarsa import SARSAAgent

# Test different hyperparameters
learning_rates = [0.1, 0.3, 0.5]
discount_factors = [0.9, 0.95, 0.99]

episodes = 3000
max_steps = 300

print("Hyperparameter Tuning for SARSA\n")

results = []

for lr in learning_rates:
    for gamma in discount_factors:
        print(f"\nTesting: Learning Rate = {lr}, Discount = {gamma}")
        
        env = BuildingEnv(grid_size=(8, 8), fire_spread_prob=0.1)
        agent = SARSAAgent(state_size=5000, action_size=5, learning_rate=lr, discount=gamma)
        
        success_count = 0
        total_rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            action = agent.choose_action(state)
            episode_reward = 0
            
            for step in range(max_steps):
                next_state, reward, done, info = env.step(action)
                next_action = agent.choose_action(next_state)
                agent.learn(state, action, reward, next_state, next_action, done)
                episode_reward += reward
                state = next_state
                action = next_action
                
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
            'avg_reward': avg_reward
        })
        
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Avg Reward: {avg_reward:.2f}")

# Print summary table
print("SUMMARY TABLE - SARSA")
print(f"{'LR':<8} {'Gamma':<8} {'Success %':<12} {'Avg Reward':<12}")

for r in results:
    print(f"{r['lr']:<8} {r['gamma']:<8} {r['success_rate']:<12.1f} {r['avg_reward']:<12.2f}")

# Find best
best = max(results, key=lambda x: x['success_rate'])
print(f"Best: LR={best['lr']}, Gamma={best['gamma']} → {best['success_rate']:.1f}% success")