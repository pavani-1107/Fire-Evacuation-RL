import matplotlib
matplotlib.use('Agg')

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os

from env.building_env import BuildingEnv
from agents.qlearning import QLearningAgent

def create_frame(grid, agent_pos, step_num, rows, cols):
    """Create a single frame of the animation."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Draw grid
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 0:  # Empty
                color = 'lightgray'
            elif grid[i, j] == 1:  # Wall
                color = 'black'
            elif grid[i, j] == 2:  # Fire
                color = 'orangered'
            elif grid[i, j] == 3:  # Exit
                color = 'limegreen'
            else:
                color = 'white'
            
            rect = Rectangle((j, rows-i-1), 1, 1, facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
    
    # Draw agent
    ax.plot(agent_pos[1] + 0.5, rows - agent_pos[0] - 0.5, 
            'bo', markersize=30, markeredgecolor='white', markeredgewidth=3)
    
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_title(f'Fire Evacuation - Step {step_num}', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Save frame
    plt.tight_layout()
    plt.savefig(f'frame_{step_num:03d}.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    return f'frame_{step_num:03d}.png'

# Train agent
print("Training agent...\n")
env = BuildingEnv(grid_size=(5, 5), fire_spread_prob=0.1)
agent = QLearningAgent(state_size=1000, action_size=5, learning_rate=0.1, discount=0.99)

for episode in range(300):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        if done:
            break

print("Creating animation...\n")

# Run episode and capture frames
state = env.reset()
frames = []
step = 0
done = False

frame_file = create_frame(env.grid, env.agent_pos, step, env.rows, env.cols)
frames.append(frame_file)

while not done and step < 50:
    action = agent.choose_action(state)
    next_state, reward, done, info = env.step(action)
    state = next_state
    step += 1
    
    frame_file = create_frame(env.grid, env.agent_pos, step, env.rows, env.cols)
    frames.append(frame_file)

# Create GIF
print("Saving GIF...")
images = [Image.open(frame) for frame in frames]
images[0].save('evacuation_animation.gif', 
               save_all=True, 
               append_images=images[1:], 
               duration=500,  # 500ms per frame
               loop=0)

# Cleanup frame files
for frame in frames:
    os.remove(frame)

print("✅ Animation saved as evacuation_animation.gif")
print(f"Total steps: {step}")