"""
reentry_env.py
--------------
Custom Gymnasium environment for hypersonic reentry guidance.

The agent controls the bank angle (sigma) at each timestep.
Goal: reach target downrange within ±50 km while satisfying
      all path constraints (heat rate, g-load, dynamic pressure).

Observation space:
    [h_norm, v_norm, gamma_norm, r_norm, q_dot_norm, n_load_norm]

Action space:
    sigma in [-pi/2, pi/2]  (bank angle, continuous)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dynamics.reentry_dynamics import (
    VehicleParams, simulate_trajectory, reentry_eom,
    exponential_atmosphere, aero_forces,
    stagnation_heat_rate, dynamic_pressure, normal_load_factor,
)

# ─── Mission & Constraint Limits ─────────────────────────────────────────────

QDOT_LIMIT  = 1.5e6     # W/m^2   (1500 kW/m^2)
NLOAD_LIMIT = 4.0       # g
QDYN_LIMIT  = 50e3      # Pa      (50 kPa)
H_TERMINAL  = 5_000.0   # m       (end of guidance phase)
R_TARGET    = 8_000e3   # m       (8000 km downrange target)
R_TOL       = 50e3      # m       (±50 km acceptable miss)


class ReentryEnv(gym.Env):
    """
    Hypersonic reentry environment.

    The agent commands bank angle sigma at each 2-second guidance step.
    The dynamics integrate forward using RK4 with the commanded sigma.

    Reward structure:
        - Positive reward for staying inside constraint corridor
        - Large penalty for constraint violations
        - Terminal reward based on downrange miss distance
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 vehicle: VehicleParams = None,
                 dt_guidance: float = 2.0,
                 noise_std: float = 0.0,
                 render_mode=None):
        super().__init__()

        self.vehicle      = vehicle or VehicleParams()
        self.dt_guidance  = dt_guidance    # seconds per guidance step
        self.noise_std    = noise_std      # sensor noise std (normalized)
        self.render_mode  = render_mode

        # ── Action space: bank angle [-90°, +90°]
        self.action_space = spaces.Box(
            low  = np.array([-np.pi / 2], dtype=np.float32),
            high = np.array([ np.pi / 2], dtype=np.float32),
        )

        # ── Observation space: 6 normalized states
        self.observation_space = spaces.Box(
            low  = -np.ones(6, dtype=np.float32),
            high =  np.ones(6, dtype=np.float32),
        )

        # Normalization references
        self._h_ref     = 120_000.0     # m
        self._v_ref     = 7_800.0       # m/s
        self._g_ref     = np.radians(10.0)
        self._r_ref     = R_TARGET
        self._qdot_ref  = QDOT_LIMIT
        self._nload_ref = QDYN_LIMIT

        self.state      = None
        self.t          = None
        self._traj_log  = None   # for rendering / plotting

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        h, v, gamma, r = self.state
        rho, _  = exponential_atmosphere(h)
        D, L    = aero_forces(v, h, self.vehicle)
        qdot    = stagnation_heat_rate(rho, v, self.vehicle.rn)
        qdyn    = dynamic_pressure(rho, v)
        nload   = normal_load_factor(L, D, self.vehicle.mass, gamma)

        obs = np.array([
            h     / self._h_ref,
            v     / self._v_ref,
            gamma / self._g_ref,
            r     / self._r_ref,
            qdot  / self._qdot_ref,
            nload / self._nload_ref,
        ], dtype=np.float32)

        # Add sensor noise (domain randomization)
        if self.noise_std > 0:
            obs += self.np_random.normal(0, self.noise_std, obs.shape).astype(np.float32)

        return np.clip(obs, -3.0, 3.0)

    def _compute_constraints(self) -> dict:
        h, v, gamma, r = self.state
        rho, _ = exponential_atmosphere(h)
        D, L   = aero_forces(v, h, self.vehicle)
        return {
            'qdot':  stagnation_heat_rate(rho, v, self.vehicle.rn),
            'qdyn':  dynamic_pressure(rho, v),
            'nload': normal_load_factor(L, D, self.vehicle.mass, gamma),
        }

    def _rk4_step(self, sigma: float) -> np.ndarray:
        """Single RK4 step for dt_guidance seconds."""
        dt  = self.dt_guidance
        s   = self.state.copy()
        veh = self.vehicle

        k1 = reentry_eom(self.t,        s,              sigma, veh)
        k2 = reentry_eom(self.t + dt/2, s + dt/2 * k1, sigma, veh)
        k3 = reentry_eom(self.t + dt/2, s + dt/2 * k2, sigma, veh)
        k4 = reentry_eom(self.t + dt,   s + dt   * k3, sigma, veh)

        return s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomize initial conditions slightly (domain randomization)
        h0     = 120_000.0 + self.np_random.uniform(-2000, 2000)
        v0     = 7_800.0   + self.np_random.uniform(-200, 200)
        gamma0 = np.radians(-5.5 + self.np_random.uniform(-0.5, 0.5))

        self.state     = np.array([h0, v0, gamma0, 0.0])
        self.t         = 0.0
        self._traj_log = [self.state.copy()]

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        sigma = float(np.clip(action[0], -np.pi/2, np.pi/2))

        # Integrate one guidance step
        new_state = self._rk4_step(sigma)
        self.state = new_state
        self.t    += self.dt_guidance
        self._traj_log.append(new_state.copy())

        h, v, gamma, r = self.state
        constraints = self._compute_constraints()

        # ── Termination conditions
        terminated = h <= H_TERMINAL or v < 500.0 or h > 130_000.0
        truncated  = self.t > 4000.0    # max episode length

        # ── Reward shaping
        reward = 0.1    # small survival reward each step

        # Constraint violation penalties
        if constraints['qdot']  > QDOT_LIMIT:
            reward -= 2.0 * (constraints['qdot'] / QDOT_LIMIT - 1.0)
        if constraints['nload'] > NLOAD_LIMIT:
            reward -= 2.0 * (constraints['nload'] / NLOAD_LIMIT - 1.0)
        if constraints['qdyn']  > QDYN_LIMIT:
            reward -= 1.0 * (constraints['qdyn'] / QDYN_LIMIT - 1.0)

        # Terminal reward: miss distance from target
        if terminated:
            miss = abs(r - R_TARGET)
            if miss < R_TOL:
                reward += 50.0 - 50.0 * (miss / R_TOL)   # up to +50
            else:
                reward -= 20.0 * (miss / R_TOL)           # penalize large miss

        info = {
            't':     self.t,
            'h':     h,
            'v':     v,
            'gamma': gamma,
            'r':     r,
            **constraints
        }

        return self._get_obs(), reward, terminated, truncated, info

    def get_trajectory_log(self) -> np.ndarray:
        """Return recorded state history as (N, 4) array."""
        return np.array(self._traj_log)

    def render(self):
        if self.render_mode == "human":
            h, v, gamma, r = self.state
            print(f"t={self.t:.0f}s | h={h/1e3:.1f}km | v={v:.0f}m/s | "
                  f"γ={np.degrees(gamma):.2f}° | r={r/1e3:.0f}km")


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = ReentryEnv(noise_std=0.01)
    obs, _ = env.reset(seed=42)
    print("Initial obs:", obs)

    total_reward = 0
    for step in range(500):
        action = env.action_space.sample()    # random policy
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step % 50 == 0:
            env.render()
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}: terminated={terminated}, truncated={truncated}")
            print(f"Final: h={info['h']/1e3:.1f} km, r={info['r']/1e3:.0f} km")
            print(f"Total reward: {total_reward:.2f}")
            break

    env.close()
