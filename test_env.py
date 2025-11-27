from env.building_env import BuildingEnv

#create environment
env = BuildingEnv(grid_size=(5, 5))

#reset and print initial state
state = env.reset()
print("Initital State: ", state)
print("Agent Position: ", env.agent_pos)
print("\nGrid")
print(env.grid)

#Take a few steps
for i in range(3):
    action = 3 #move right
    state, reward, done, info = env.step(action)
    print(f"\nStep {i + 1}: Action = Right, Reward = {reward}, Done = {done}")
    print("State: ", state)

print("\nEnvironment Works")