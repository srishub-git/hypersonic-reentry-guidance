"""
train_ppo.py
------------
Train a PPO agent on the ReentryEnv.
Run from the project root:
    python training/train_ppo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs.reentry_env import ReentryEnv

# ── Paths
RESULTS_DIR   = "results"
MODEL_DIR     = os.path.join(RESULTS_DIR, "models")
LOG_DIR       = os.path.join(RESULTS_DIR, "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

# ── Hyperparameters
N_ENVS        = 4           # parallel environments
TOTAL_STEPS   = 2_000_000  # increase to 3M for best results
EVAL_FREQ     = 20_000
N_EVAL_EPS    = 10
R_TARGET = 1_000e3

def make_train_env():
    """Training env with noise (domain randomization)."""
    return Monitor(ReentryEnv(noise_std=0.02))

def make_eval_env():
    """Clean eval env without noise."""
    return Monitor(ReentryEnv(noise_std=0.0))


if __name__ == "__main__":
    print("=" * 60)
    print("  Hypersonic Reentry Guidance — PPO Training")
    print("=" * 60)

    # ── Vectorized training environments
    train_env = make_vec_env(make_train_env, n_envs=N_ENVS)
    eval_env  = make_vec_env(make_eval_env,  n_envs=1)

    # ── Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = MODEL_DIR,
        log_path             = LOG_DIR,
        eval_freq            = EVAL_FREQ // N_ENVS,
        n_eval_episodes      = N_EVAL_EPS,
        deterministic        = True,
        verbose              = 1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq  = 100_000 // N_ENVS,
        save_path  = MODEL_DIR,
        name_prefix= "reentry_ppo",
    )

    # ── PPO model
    # Network: two hidden layers of 128 neurons each
    policy_kwargs = dict(net_arch=[128, 128])

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs  = policy_kwargs,
        learning_rate  = 3e-4,
        n_steps        = 2048,
        batch_size     = 256,
        n_epochs       = 10,
        gamma          = 0.995,          # high gamma -- long-horizon task
        gae_lambda     = 0.95,
        clip_range     = 0.2,
        ent_coef       = 0.005,          # small entropy bonus
        vf_coef        = 0.5,
        max_grad_norm  = 0.5,
        verbose        = 1,
        tensorboard_log= LOG_DIR,
        device         = "auto",
    )

    print(f"\nTraining for {TOTAL_STEPS:,} steps across {N_ENVS} envs...")
    print(f"Monitor training:  tensorboard --logdir {LOG_DIR}\n")

    model.learn(
        total_timesteps = TOTAL_STEPS,
        callback        = [eval_callback, checkpoint_callback],
        progress_bar    = True,
    )

    # Save final model
    final_path = os.path.join(MODEL_DIR, "reentry_ppo_final")
    model.save(final_path)
    print(f"\nFinal model saved to {final_path}")

    # ── Quick evaluation
    print("\nRunning final evaluation (10 episodes)...")
    model = PPO.load(os.path.join(MODEL_DIR, "best_model"))

    rewards, misses = [], []
    for ep in range(10):
        obs, _ = eval_env.envs[0].reset()
        ep_reward, done = 0, False
        while not done:
            action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.envs[0].step(action[0])
            ep_reward += reward
            done = terminated or truncated
        miss = abs(info['r'] - R_TARGET) / 1e3
        print(f"  Ep {ep + 1:2d}: reward={ep_reward:7.1f} | miss={miss:.1f} km | h_final={info['h'] / 1e3:.1f} km")
        rewards.append(ep_reward)
        misses.append(miss)

    print(f"\nMean reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    print(f"Mean miss:   {np.mean(misses):.1f} +/- {np.std(misses):.1f} km")
