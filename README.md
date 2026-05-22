# Hypersonic Reentry Guidance through Hybrid Optimal Control & Reinforcement Learning

> A graduate level personal project implementing reference trajectory generation and RL based guidance for a hypersonic reentry vehicle under aerothermal, g-load, and dynamic pressure constraints.

---

## Motivation

Hypersonic reentry guidance is one of the most constrained trajectory optimization problems in aerospace engineering. The vehicle must:
- Survive extreme aerodynamic heating (heat rate and heat load limits)
- Keep structural g-loads within human/payload tolerance
- Hit a precise landing target (cross-range and downrange)
- Do all of this with significant model uncertainty

Classical approaches (Predictor-Corrector, Apollo) work but are brittle to uncertainty. This project combines **optimal control** (reference trajectory generation) with **deep RL (PPO)** to learn a robust guidance policy that tracks the reference under simulated sensor noise and atmospheric uncertainty.

---

## Project Structure

```
hypersonic-reentry-guidance/
├── dynamics/
│   └── reentry_dynamics.py        # 3DOF equations of motion
├── envs/
│   └── reentry_env.py             # Custom Gymnasium environment
├── reference/
│   └── predictor_corrector.py     # Classical baseline guidance
├── training/
│   └── train_ppo.py               # PPO training script
├── results/                       # Saved models, plots
├── notebooks/
│   └── 01_dynamics_exploration.ipynb
├── requirements.txt
└── README.md
```

---

## Vehicle & Mission Parameters

| Parameter | Value |
|---|---|
| Vehicle | Generic lifting body (Apollo-class) |
| Entry altitude | 120 km |
| Entry velocity | 7,800 m/s |
| Entry flight path angle | -5.5° |
| Ballistic coefficient | ~100 kg/m² |
| Target downrange | 8,000 km |

---

## Constraints

| Constraint | Limit |
|---|---|
| Stagnation heat rate | ≤ 1500 kW/m² |
| Total heat load | ≤ 300 MJ/m² |
| Normal g-load | ≤ 4 g |
| Dynamic pressure | ≤ 50 kPa |



## References

1. Shen & Lu (2003) — *Onboard Generation of Three-Dimensional Constrained Entry Trajectories*
2. Vinh et al. — *Hypersonic and Planetary Entry Flight Mechanics* (textbook)
3. Schulman et al. (2017) — *Proximal Policy Optimization Algorithms*
4. Chapman (1958) — *An Approximate Analytical Method for Studying Entry into Planetary Atmospheres*

