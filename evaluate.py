import gymnasium as gym
from stable_baselines3 import PPO
import time

# Create environment with rendering enabled
env = gym.make("CarRacing-v3", render_mode="human")
model = PPO.load("models/ppo_carracing")

# Evaluate the agent for a few episodes
rewards = []
for episode in range(10):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    rewards.append(total_reward)

print(f"Average Reward: {sum(rewards)/len(rewards):.2f}")
print(f"Max Reward: {max(rewards):.2f}")
print(f"Min Reward: {min(rewards):.2f}")


env.close()
