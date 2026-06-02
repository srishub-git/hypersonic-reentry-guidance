"""
comparison_plot.py
------------------
Compare predictor-corrector vs PPO agent under atmospheric uncertainty.
Runs both methods across density perturbations and plots miss distance.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from dynamics.reentry_dynamics import VehicleParams
from reference.predictor_corrector import corrector, predict_landing
from envs.reentry_env import ReentryEnv
import dynamics.reentry_dynamics as dyn

# ── Settings
R_TARGET     = 1_000e3
PERTURBATIONS = [-0.20, -0.10, 0.0, 0.10, 0.20]
N_EPISODES   = 10      # PPO episodes per perturbation level
MODEL_PATH   = "results/models/best_model"

vehicle = VehicleParams()

state0 = np.array([
    120_000.0,
    7_800.0,
    np.radians(-5.5),
    0.0
])


def evaluate_predictor_corrector(perturbations):
    """Run predictor-corrector across density perturbations."""
    misses = []

    for pert in perturbations:
        # Perturb atmosphere
        original = dyn.RHO_SL
        dyn.RHO_SL = original * (1 + pert)

        # Run corrector
        sigma = corrector(state0, vehicle)
        r_pred = predict_landing(state0, sigma, vehicle)
        miss = abs(r_pred - R_TARGET) / 1e3
        misses.append(miss)

        print(f"  PC | density {pert * 100:+.0f}%: miss={miss:.1f} km")

        # Restore atmosphere
        dyn.RHO_SL = original

    return misses


def evaluate_ppo(perturbations, n_episodes):
    """Run PPO agent across density perturbations."""
    model = PPO.load(MODEL_PATH)
    mean_misses = []

    for pert in perturbations:
        original = dyn.RHO_SL
        dyn.RHO_SL = original * (1 + pert)

        episode_misses = []
        for ep in range(n_episodes):
            env = ReentryEnv(noise_std=0.0)
            obs, _ = env.reset(seed=ep)
            done = False

            while not done:
                action, _ = model.predict(
                    obs.reshape(1, -1),
                    deterministic=True
                )
                obs, _, terminated, truncated, info = env.step(action[0])
                done = terminated or truncated

            miss = abs(info['r'] - R_TARGET) / 1e3
            episode_misses.append(miss)

        mean_miss = np.mean(episode_misses)
        mean_misses.append(mean_miss)
        print(f"  PPO | density {pert * 100:+.0f}%: mean miss={mean_miss:.1f} km")

        dyn.RHO_SL = original

    return mean_misses


def plot_comparison(perturbations, pc_misses, ppo_misses):
    """Plot miss distance comparison."""
    perts_pct = [p * 100 for p in perturbations]

    plt.figure(figsize=(10, 6))

    plt.plot(perts_pct, pc_misses, 'b-o',
             linewidth=2, markersize=8, label='Predictor-Corrector')
    plt.plot(perts_pct, ppo_misses, 'r-o',
             linewidth=2, markersize=8, label='PPO Agent')
    plt.axhline(y=20, color='g', linestyle='--',
                linewidth=2, label='20 km Tolerance')

    plt.xlabel('Atmospheric Density Perturbation (%)', fontsize=12)
    plt.ylabel('Miss Distance (km)', fontsize=12)
    plt.title('Guidance Performance Under Atmospheric Uncertainty', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(perts_pct)

    plt.tight_layout()
    plt.savefig('results/comparison_plot.png', dpi=150)
    plt.show()
    print("Plot saved to results/comparison_plot.png")


if __name__ == "__main__":
    print("Evaluating Predictor-Corrector...")
    pc_misses = evaluate_predictor_corrector(PERTURBATIONS)

    print("\nEvaluating PPO Agent...")
    ppo_misses = evaluate_ppo(PERTURBATIONS, N_EPISODES)

    print("\nResults Summary:")
    print(f"{'Perturbation':>15} | {'PC Miss (km)':>12} | {'PPO Miss (km)':>13}")
    print("-" * 45)
    for p, pc, ppo in zip(PERTURBATIONS, pc_misses, ppo_misses):
        print(f"{p * 100:>14.0f}% | {pc:>12.1f} | {ppo:>13.1f}")

    plot_comparison(PERTURBATIONS, pc_misses, ppo_misses)