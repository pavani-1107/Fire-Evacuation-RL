from env.building_env import BuildingEnv

print("Testing Simple Environment (5x5):\n")
env = BuildingEnv(grid_size=(8, 8), complex_layout=False)
state = env.reset()
print("Grid:")
print(env.grid)
print(f"Agent position: {env.agent_pos}")
print(f"State: {state}\n")

print("=" * 50)

print("\nTesting Complex Environment (10x10):\n")
env_complex = BuildingEnv(grid_size=(10, 10), complex_layout=True)
state = env_complex.reset()
print("Grid:")
print(env_complex.grid)
print(f"Agent position: {env_complex.agent_pos}")
print(f"State: {state}")

print("\nBoth environments work!")