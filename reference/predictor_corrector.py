"""
predictor_corrector.py
----------------------

Classical predictor-corrector guidance for hypersonic reentry.

Every guidance cycle:
    1. Predict landing location using current bank angle
    2. Compare predicted downrange to target
    3. Correct bank angle using binary search until error < 50km

I am listening to The Scientist while coding cuz its an awesome song (DELETE LINE LATER!!!!!!!)

"""

import numpy as np
from dynamics.reentry_dynamics import (
    VehicleParams,
    simulate_trajectory,    # this function is for simulating the physics forward in time and returns where the vehicle ends up
    exponential_atmosphere,
)

# Mission Target
R_TARGET = 8_000e3      # 8000 km in meters
R_TOL    = 50e3         # acceptable miss distance (50 km)
H_TERMINAL = 5_000.0   # guidance ends at 5 km altitude

"""
this is the limit because above this vehicle is still moving fast enough that aerodynamic bank angle still works.

below 5km atmosphere is thick, speeds are slow, and lift barely works and we are seconds away from landing

parachute should take from here
"""


# PREDICTOR FUNCTION

def predict_landing(state: np.ndarray,                      # current state [4 EOMs]
                    sigma: float,                           # bank angle to test
                    vehicle: VehicleParams) -> float:       # vehicle parameters
    """
    This simulates forward from current state with fixed bank angle.
    Returns predicted downrange distance at terminal altitude.
    """
    def constant_bank(t, s):
        return sigma

    traj = simulate_trajectory(
        state,
        constant_bank,
        vehicle,
        t_span=(0, 4000),       # fixed: added missing comma here
        h_terminal=H_TERMINAL
    )

    return traj['r'][-1]


# CORRECTOR FUNCTION

def corrector(state: np.ndarray,                            # current state [4 EOMs]
              vehicle: VehicleParams,                       # vehicle parameters
              sigma_low: float = 0.0,                      # minimum bank angle to search (rad)
              sigma_high: float = np.pi / 2) -> float:     # maximum angle to search (obviously rad!!)
    """
    Binary search over bank angle to find sigma that hits R_TARGET.
    Returns sigma, the bank angle that minimizes range error.
    """
    for _ in range(20):                                     # 20 iterations should be enough to converge
        sigma_mid = (sigma_low + sigma_high) / 2.0
        r_predicted = predict_landing(state, sigma_mid, vehicle)    # fixed: was predicted_landing (typo)

        if r_predicted < R_TARGET:
            sigma_high = sigma_mid      # landing short, reduce bank angle
        else:
            sigma_low = sigma_mid       # landing long, increase bank angle

        if abs(r_predicted - R_TARGET) < R_TOL:
            break

    return sigma_mid


# CALLING THE GUIDANCE FUNCTION EVERY 2 SECONDS

def guidance_step(state: np.ndarray,                        # fixed: was npndarray (typo)
                  vehicle: VehicleParams,
                  sigma_prev: float) -> float:              # bank angle from previous step
    """
    Called every guidance cycle during reentry.
    Returns bank angle command for current timestep.
    """
    h, v, gamma, r = state

    if h > 80_000 or v < 1000:                             # fixed: was v > 1000, should be v < 1000
        return sigma_prev

    sigma_cmd = corrector(state, vehicle)

    max_rate = np.radians(5.0)                             # max 5 degrees change per step
    delta = np.clip(sigma_cmd - sigma_prev, -max_rate, max_rate)

    return sigma_prev + delta


# TEST

if __name__ == "__main__":
    vehicle = VehicleParams()

    # Starting state
    state0 = np.array([
        120_000.0,
        7_800.0,
        np.radians(-5.5),
        0.0
    ])

    print("Testing predictor...")
    r_pred = predict_landing(state0, np.radians(30), vehicle)
    print(f"  With 30 deg bank: predicted landing = {r_pred/1e3:.0f} km")

    r_pred = predict_landing(state0, np.radians(60), vehicle)
    print(f"  With 60 deg bank: predicted landing = {r_pred/1e3:.0f} km")

    print("\nTesting corrector...")
    sigma = corrector(state0, vehicle)
    print(f"  Corrector found sigma = {np.degrees(sigma):.1f} deg")

    r_check = predict_landing(state0, sigma, vehicle)
    print(f"  Predicted landing with that sigma = {r_check/1e3:.0f} km")
    print(f"  Target = {R_TARGET/1e3:.0f} km")
    print(f"  Miss = {abs(r_check - R_TARGET)/1e3:.1f} km")