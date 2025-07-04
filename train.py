# train.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
import os

print("🚗 Starting CarRacing PPO training...")

# Create and wrap the environment
env = DummyVecEnv([lambda: Monitor(gym.make("CarRacing-v3", render_mode=None))])
env = VecTransposeImage(env)  # For CNN policy compatibility

# Create model
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=2.5e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    tensorboard_log="./logs",
)

# Train the model
model.learn(total_timesteps=1_000_000)
model.save("models/ppo_carracing")

env.close()
print("✅ Training complete and model saved.")
