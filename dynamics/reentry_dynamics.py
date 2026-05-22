"""
reentry_dynamics.py
--------------------
3DOF Equations of Motion for hypersonic reentry over a spherical, rotating Earth.

State vector:
    h   : altitude (m)
    v   : velocity magnitude (m/s)
    gamma : flight path angle (rad)  -- negative = descending
    r   : downrange distance (m)

Control:
    sigma : bank angle (rad) -- primary control for lift direction

Reference:
    Vinh et al., "Hypersonic and Planetary Entry Flight Mechanics", Ch. 3
"""

import numpy as np
from dataclasses import dataclass


# ─── Planet & Atmosphere Constants ───────────────────────────────────────────

R_EARTH   = 6.3781e6       # Earth radius (m)
MU_EARTH  = 3.986e14       # Gravitational parameter (m^3/s^2)
OMEGA_E   = 7.2921e-5      # Earth rotation rate (rad/s)
RHO_SL    = 1.225          # Sea-level density (kg/m^3)
H_SCALE   = 7254.0         # Density scale height (m)  -- exponential atmosphere
G0        = 9.80665        # Standard gravity (m/s^2)


# ─── Vehicle Parameters ───────────────────────────────────────────────────────

@dataclass
class VehicleParams:
    """Aerodynamic and physical parameters of the reentry vehicle."""
    mass:     float = 2800.0      # kg   (Apollo-class)
    area_ref: float = 12.0        # m^2  (reference area)
    CD:       float = 1.2         # Drag coefficient (hypersonic, blunt body)
    CL:       float = 0.3         # Lift coefficient
    rn:       float = 0.3         # Nose radius (m) -- for heating calc

    @property
    def beta(self) -> float:
        """Ballistic coefficient (kg/m^2)."""
        return self.mass / (self.CD * self.area_ref)

    @property
    def LD(self) -> float:
        """Lift-to-drag ratio."""
        return self.CL / self.CD


# ─── Atmosphere Model ─────────────────────────────────────────────────────────

def exponential_atmosphere(h: float) -> tuple[float, float]:
    """
    Exponential atmosphere model.

    Parameters
    ----------
    h : altitude (m)

    Returns
    -------
    rho : density (kg/m^3)
    a   : speed of sound (m/s)  -- rough approximation
    """
    h = max(h, 0.0)
    rho = RHO_SL * np.exp(-h / H_SCALE)

    # Rough piecewise speed of sound
    if h < 11000:
        T = 288.15 - 0.0065 * h
    elif h < 25000:
        T = 216.65
    else:
        T = 216.65 + 0.003 * (h - 25000)
    T = max(T, 180.0)
    a = np.sqrt(1.4 * 287.05 * T)

    return rho, a


# ─── Aerodynamic Forces ───────────────────────────────────────────────────────

def aero_forces(v: float, h: float, vehicle: VehicleParams) -> tuple[float, float]:
    """
    Compute drag (D) and lift (L) in Newtons.

    Parameters
    ----------
    v       : velocity (m/s)
    h       : altitude (m)
    vehicle : VehicleParams

    Returns
    -------
    D : drag force (N)
    L : lift force (N)
    """
    rho, _ = exponential_atmosphere(h)
    q = 0.5 * rho * v**2          # dynamic pressure (Pa)
    D = q * vehicle.area_ref * vehicle.CD
    L = q * vehicle.area_ref * vehicle.CL
    return D, L


# ─── Constraint Functions ─────────────────────────────────────────────────────

def stagnation_heat_rate(rho: float, v: float, rn: float) -> float:
    """
    Chapman's formula for stagnation-point convective heat rate.

        q_dot = k * sqrt(rho / rn) * v^3

    Chapman, D.R. (1958). k ≈ 1.83e-4 (SI units, q in W/m^2)

    Parameters
    ----------
    rho : density (kg/m^3)
    v   : velocity (m/s)
    rn  : nose radius (m)

    Returns
    -------
    q_dot : heat rate (W/m^2)
    """
    k = 1.83e-4
    return k * np.sqrt(rho / rn) * v**3


def dynamic_pressure(rho: float, v: float) -> float:
    """q = 0.5 * rho * v^2  (Pa)"""
    return 0.5 * rho * v**2


def normal_load_factor(L: float, D: float, mass: float, gamma: float) -> float:
    """
    Approximate normal g-load experienced by vehicle/crew.
    n = sqrt(L^2 + D^2) / (m * g0)  -- simplified
    """
    return np.sqrt(L**2 + D**2) / (mass * G0)


# ─── Equations of Motion ─────────────────────────────────────────────────────

def reentry_eom(t: float,
                state: np.ndarray,
                sigma: float,
                vehicle: VehicleParams) -> np.ndarray:
    """
    3DOF equations of motion for planar reentry (range-altitude plane).

    State: [h, v, gamma, r]
        h     : altitude (m)
        v     : velocity (m/s)
        gamma : flight path angle (rad)
        r     : downrange distance (m)

    Control:
        sigma : bank angle (rad)

    Returns
    -------
    dstate/dt : np.ndarray of shape (4,)
    """
    h, v, gamma, r = state

    # Gravity at altitude
    g = MU_EARTH / (R_EARTH + h)**2

    # Atmosphere & aero
    rho, _ = exponential_atmosphere(h)
    D, L = aero_forces(v, h, vehicle)

    # Lift & drag accelerations
    aD = D / vehicle.mass
    aL = L / vehicle.mass

    # EOM (non-rotating spherical Earth, planar)
    dh     = v * np.sin(gamma)
    dv     = -aD - g * np.sin(gamma)
    dgamma = (1.0 / v) * (aL * np.cos(sigma) - (g - v**2 / (R_EARTH + h)) * np.cos(gamma))
    dr     = (R_EARTH / (R_EARTH + h)) * v * np.cos(gamma)

    return np.array([dh, dv, dgamma, dr])


# ─── Trajectory Simulator ─────────────────────────────────────────────────────

def simulate_trajectory(state0: np.ndarray,
                        sigma_func,
                        vehicle: VehicleParams,
                        t_span: tuple = (0, 3000),
                        dt: float = 0.5,
                        h_terminal: float = 5000.0) -> dict:
    """
    Integrate reentry trajectory using RK4.

    Parameters
    ----------
    state0     : initial state [h0, v0, gamma0, r0]
    sigma_func : callable(t, state) -> bank angle (rad)
    vehicle    : VehicleParams
    t_span     : (t_start, t_end) in seconds
    dt         : timestep (s)
    h_terminal : stop integration at this altitude (m)

    Returns
    -------
    dict with keys: t, h, v, gamma, r, q_dot, q_dyn, n_load
    """
    from scipy.integrate import solve_ivp

    # Wrapper for solve_ivp
    def ode(t, state):
        sigma = sigma_func(t, state)
        return reentry_eom(t, state, sigma, vehicle)

    # Terminal event: hit target altitude
    def hit_ground(t, state):
        return state[0] - h_terminal
    hit_ground.terminal  = True
    hit_ground.direction = -1

    sol = solve_ivp(
        ode,
        t_span,
        state0,
        method='RK45',
        max_step=dt,
        events=hit_ground,
        dense_output=False,
        rtol=1e-6,
        atol=1e-8
    )

    # Unpack
    t      = sol.t
    h      = sol.y[0]
    v      = sol.y[1]
    gamma  = sol.y[2]
    r      = sol.y[3]

    # Derived quantities
    q_dot   = np.zeros_like(t)
    q_dyn   = np.zeros_like(t)
    n_load  = np.zeros_like(t)

    for i in range(len(t)):
        rho, _ = exponential_atmosphere(h[i])
        D_i, L_i = aero_forces(v[i], h[i], vehicle)
        q_dot[i]  = stagnation_heat_rate(rho, v[i], vehicle.rn)
        q_dyn[i]  = dynamic_pressure(rho, v[i])
        n_load[i] = normal_load_factor(L_i, D_i, vehicle.mass, gamma[i])

    return {
        't':      t,
        'h':      h,
        'v':      v,
        'gamma':  gamma,
        'r':      r,
        'q_dot':  q_dot,       # W/m^2
        'q_dyn':  q_dyn,       # Pa
        'n_load': n_load,      # g
    }


# ─── Quick Sanity Check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    vehicle = VehicleParams()
    print(f"Vehicle: beta={vehicle.beta:.1f} kg/m², L/D={vehicle.LD:.2f}")

    # Initial conditions: 120 km altitude, 7800 m/s, -5.5 deg FPA
    state0 = np.array([
        120_000.0,          # h (m)
        7_800.0,            # v (m/s)
        np.radians(-5.5),   # gamma (rad)
        0.0                 # r (m)
    ])

    # Constant zero bank angle (ballistic)
    def ballistic(t, state):
        return 0.0

    print("Simulating ballistic trajectory...")
    traj = simulate_trajectory(state0, ballistic, vehicle, t_span=(0, 2500))

    print(f"  Final altitude : {traj['h'][-1]/1000:.1f} km")
    print(f"  Final velocity : {traj['v'][-1]:.0f} m/s")
    print(f"  Downrange      : {traj['r'][-1]/1000:.0f} km")
    print(f"  Peak heat rate : {traj['q_dot'].max()/1e3:.0f} kW/m²")
    print(f"  Peak g-load    : {traj['n_load'].max():.2f} g")
    print(f"  Peak dyn press : {traj['q_dyn'].max()/1e3:.1f} kPa")

    # ── Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Hypersonic Reentry — Ballistic Trajectory (Sanity Check)", fontsize=13)

    axes[0,0].plot(traj['r']/1e3, traj['h']/1e3)
    axes[0,0].set_xlabel("Downrange (km)"); axes[0,0].set_ylabel("Altitude (km)")
    axes[0,0].set_title("Altitude vs Downrange"); axes[0,0].grid(True)

    axes[0,1].plot(traj['t'], traj['v']/1e3)
    axes[0,1].set_xlabel("Time (s)"); axes[0,1].set_ylabel("Velocity (km/s)")
    axes[0,1].set_title("Velocity"); axes[0,1].grid(True)

    axes[0,2].plot(traj['t'], np.degrees(traj['gamma']))
    axes[0,2].set_xlabel("Time (s)"); axes[0,2].set_ylabel("FPA (deg)")
    axes[0,2].set_title("Flight Path Angle"); axes[0,2].grid(True)

    axes[1,0].plot(traj['t'], traj['q_dot']/1e3, color='red')
    axes[1,0].axhline(1500, color='red', linestyle='--', label='Limit 1500 kW/m²')
    axes[1,0].set_xlabel("Time (s)"); axes[1,0].set_ylabel("Heat Rate (kW/m²)")
    axes[1,0].set_title("Stagnation Heat Rate"); axes[1,0].legend(); axes[1,0].grid(True)

    axes[1,1].plot(traj['t'], traj['n_load'], color='orange')
    axes[1,1].axhline(4.0, color='orange', linestyle='--', label='Limit 4 g')
    axes[1,1].set_xlabel("Time (s)"); axes[1,1].set_ylabel("Normal Load (g)")
    axes[1,1].set_title("G-Load"); axes[1,1].legend(); axes[1,1].grid(True)

    axes[1,2].plot(traj['t'], traj['q_dyn']/1e3, color='green')
    axes[1,2].axhline(50, color='green', linestyle='--', label='Limit 50 kPa')
    axes[1,2].set_xlabel("Time (s)"); axes[1,2].set_ylabel("Dynamic Pressure (kPa)")
    axes[1,2].set_title("Dynamic Pressure"); axes[1,2].legend(); axes[1,2].grid(True)

    plt.tight_layout()
    plt.savefig("results/ballistic_sanity_check.png", dpi=150)
    plt.show()
    print("Plot saved to results/ballistic_sanity_check.png")
