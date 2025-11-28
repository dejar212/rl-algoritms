from env.grid_world import GridWorldEnv
from env.ma_wrapper import CentralizedWrapper

print("Checking Mixed Environment...")
env_raw = GridWorldEnv(width=50, height=50, n_agents=5, fov_radius=5, obs_mode='mixed')
env = CentralizedWrapper(env_raw)

obs, info = env.reset()

print(f"Observation Type: {type(obs)}")
for key, value in obs.items():
    print(f"Key: {key}, Shape: {value.shape}, Type: {value.dtype}")
    print(f"Sample Data: {value.flatten()[:5]}")

assert obs['image'].shape == (15, 11, 11), f"Wrong image shape: {obs['image'].shape}"
assert obs['vector'].shape == (10,), f"Wrong vector shape: {obs['vector'].shape}"

print("Check Passed!")
env.close()

