# Hypersonic Reentry Guidance through Hybrid Optimal Control and Reinforcement Learning

This project investigates guidance strategies for hypersonic reentry vehicles under strict path constraints. The core idea is to combine classical optimal control used to generate a reference trajectory with a deep reinforcement learning policy (PPO) that learns to track that reference under atmospheric uncertainty and sensor noise.

The motivation comes from the fact that classical predictor-corrector guidance methods, while reliable, are weak to dispersions in atmospheric density and vehicle aerodynamics. A learned policy trained with domain randomization can potentially generalize better across these uncertainties while still respecting hard physical constraints on heating, g-load, and dynamic pressure.

---

## Problem Statement

A lifting reentry vehicle enters the atmosphere at approximately 7,800 m/s from 120 km altitude (NASA.gov). The guidance system must modulate the bank angle to control the lift vector, shaping the trajectory to hit a downrange target while keeping the vehicle within a safe flight corridor defined by:

- Stagnation heat rate: <= 1500 kW/m²
- Normal load factor: <= 4 g
- Dynamic pressure: <= 50 kPa

---

## Approach

The project is structured in two phases. First, a reference trajectory is generated using a predictor-corrector scheme based on drag modulation. Second, a PPO agent is trained in a custom Gymnasium environment to track this reference under randomized initial conditions and atmospheric perturbations.

---

## Repository Structure

```
hypersonic-reentry-guidance/
├── dynamics/
│   └── reentry_dynamics.py        # 3DOF equations of motion, atmosphere model, constraint functions
├── envs/
│   └── reentry_env.py             # Custom Gymnasium environment for RL training
├── reference/
│   └── predictor_corrector.py     # Classical baseline guidance (to be implemented)
├── training/
│   └── train_ppo.py               # PPO training script using Stable Baselines3
├── results/                       # Saved models, training curves, trajectory plots
├── notebooks/
│   └── 01_dynamics_exploration.ipynb
├── requirements.txt
└── README.md
```

---

## Vehicle Parameters

| Parameter | Value |
|---|---|
| Vehicle class | Apollo-class lifting body |
| Mass | 2800 kg |
| Reference area | 12 m² |
| Drag coefficient | 1.2 |
| Lift-to-drag ratio | 0.25 |
| Nose radius | 0.3 m |
| Entry altitude | 120 km |
| Entry velocity | 7,800 m/s |
| Entry flight path angle | -5.5 deg |

---



## References

1. Vinh, N.X., Busemann, A., Culp, R.D. — *Hypersonic and Planetary Entry Flight Mechanics*, University of Michigan Press
2. Shen, Z., Lu, P. (2003) — *Onboard Generation of Three-Dimensional Constrained Entry Trajectories*
3. Chapman, D.R. (1958) — *An Approximate Analytical Method for Studying Entry into Planetary Atmospheres*, NACA TN-4276
4. Schulman, J. et al. (2017) — *Proximal Policy Optimization Algorithms*, arXiv:1707.06347
