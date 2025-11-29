import matplotlib
matplotlib.use('Agg')

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os

from env.building_env import BuildingEnv
from agents.sarsa import SARSAAgent

def create_frame(grid, agent_pos, step_num, rows, cols):
    """Create a single frame of the animation."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
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
    ax.set_title(f'SARSA Fire Evacuation - Step {step_num}', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Save frame
    plt.tight_layout()
    plt.savefig(f'frame_{step_num:03d}.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    return f'frame_{step_num:03d}.png'

# Train agent
print("Training SARSA agent...\n")
env = BuildingEnv(grid_size=(8, 8), fire_spread_prob=0.1)
agent = SARSAAgent(state_size=5000, action_size=5, learning_rate=0.1, discount=0.99)

for episode in range(500):
    state = env.reset()
    action = agent.choose_action(state)
    done = False
    
    while not done:
        next_state, reward, done, info = env.step(action)
        next_action = agent.choose_action(next_state)
        agent.learn(state, action, reward, next_state, next_action, done)
        state = next_state
        action = next_action
        if done:
            break

print("Creating animation...\n")

# Run episode and capture frames
state = env.reset()
action = agent.choose_action(state)
frames = []
step = 0
done = False

frame_file = create_frame(env.grid, env.agent_pos, step, env.rows, env.cols)
frames.append(frame_file)

while not done and step < 100:
    next_state, reward, done, info = env.step(action)
    next_action = agent.choose_action(next_state)
    state = next_state
    action = next_action
    step += 1
    
    frame_file = create_frame(env.grid, env.agent_pos, step, env.rows, env.cols)
    frames.append(frame_file)

# Create GIF
print("Saving GIF...")
images = [Image.open(frame) for frame in frames]
images[0].save('sarsa_evacuation.gif', 
               save_all=True, 
               append_images=images[1:], 
               duration=500,  # 500ms per frame
               loop=0)

# Cleanup frame files
for frame in frames:
    os.remove(frame)

print("SARSA animation saved as sarsa_evacuation.gif")
print(f"Total steps: {step}")