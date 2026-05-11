##############################################################################
############## Program to simulate laser rate equations in Python ############
############################ Niall Boohan 2018 ###############################
############################## boohann@tcd.ie ################################
##############################################################################


### Theory and equations sourced from:
# Title: Extraction of DFB laser rate equation parameters for system 
#        simulation purposes
# Authors: J. C. Cartledge and R. C. Srinivasan
# DOI: 10.1109/50.580827

# Title: Extraction of laser rate equations parameters for representative
#        simulations of metropolitan-area transmission systems and network
# Authors: I. Tomkos, I. Roudas, R. Hesse, N. Antoniades, A. Boskovic,
#           R. Vodhanel
# DOI: https://doi.org/10.1016/S0030-4018(01)01230-5


# Version 2.0.0 7/5/26 ###



### Import necessary libraries ###
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import constants


### Simulate dymanic or steady-state ###
CALC_MODE = 1                               # 0=dynamic, 1 = steady-state LI 
CURRENT_DYNAMIC = 20e-3                     # Const current dynamic calc (A)
CURRENT_SWEEP = np.linspace(0, 50e-3, 100)  # Cur sweep for steady-state (A)


### Simulation input parameters dictionary ###
LASER_PARAMS = {
    'α': 5,              # Lasing mode cavity loss (cm^-1)
    'n': 3.2,            # Cavity refractive index
    'L': 400,            # Cavity length (um)
    'w': 2,              # Cavity width (um)
    'h_active': 100,     # Height active region (nm)
    'λ': 1300,           # Lasing mode wavelength (nm)
    'r_l': 0.5,          # Left facet amplitude reflectivity
    'r_r': 0.5,          # Right facet amplitude reflectivity
    'β': 1e-4,           # Spontaneous emission factor
    'Γ': 0.15,           # Quantum well confinement factor
    'τ_n': 1.0e-9,       # Carrier relaxation time (s)
    'g': 1.5e-5,         # Gain slope constant (cm^3s^-1)
    'N_tr': 1e17,        # Transparency carrier concentration (cm^-3)
    'ε': 1.5e-17,        # Gain compression factor (cm^3)
}


# Hard-coded simulation parameters ###
class SimConfig:
    T_END = 2.5e-9         # End time (s)
    DT = 1e-13           # Time step (s)
    N_INITIAL = 1e16     # Initial carrier concentration (cm^-3)
    S_INITIAL = 0        # Initial photon concentration (cm^-3)


### Helper functions ###
def calculate_cavity_volume(p):
    # Convert dimensions: L (um), w (um), h_active (nm) -> cm^3
    return p['L'] * p['w'] * p['h_active'] * (1e-15)

def calculate_photon_lifetime(p):
    # Round-trip photon time (s)
    τ_p = 1 / ((constants.c / (p['L'] * 1e-6)) * 
                np.log(1 / (p['r_l'] * p['r_r'])))
    # Photon lifetime from cavity loss (s)
    τ_α = 1 / (constants.c * p['α'] * 100)    
    return τ_p, τ_α

def calculate_mode_frequency(p):
    # Mode frequency (Hz)
    return constants.c / (p['λ'] / 1e9)


### Rate equations ###
def laser_rates(t, y, p, I):
    """Rate equations for laser carrier and photon dynamics.
    
    Solves the coupled differential equations:
    dN/dt = I/(q*V) - N/τ_n - g*S*(N-N_tr)/(1+ε*S)
    dS/dt = Γ*g*S*(N-N_tr)/(1+ε*S) - S/(τ_p+τ_α) + Γ*β*N/τ_n
    
    Args:
        t: Time (s)
        y: State vector [N, S], where N is carrier concentration,
            S is photon concentration
        p: Laser parameters dictionary
        I: Injection current (A)
    
    Returns:
        dy: Carrier and photon concentration change for 
            time-step [dN/dt, dS/dt]
    """
    dy = np.zeros(2)
    
    # Calculate cavity parameters
    τ_p, τ_α = calculate_photon_lifetime(p)
    V = calculate_cavity_volume(p)
    
    # Extract state variables
    N, S = y[0], y[1]
    
    # Carrier rate equation
    dy[0] = ((I / (constants.e*V)) - (N/p['τ_n']) - 
            p['g']*(N-p['N_tr'])*(S/(1+p['ε']*S)))
    
    # Photon rate equation
    dy[1] = ((p['Γ']*p['g']*(N - p['N_tr'])*(S/(1+p['ε']*S)) -
             S/(τ_p + τ_α) + (p['Γ']*p['β']*N)/p['τ_n']))
    
    return dy


def solve(p, I):
    """Solve laser rate equations using Runge-Kutta integration.
    Args:
        p: Laser parameters dictionary
        I: Injection current (A)
    
    Returns:
        T: Time array (s)
        N: Carrier concentration array (cm^-3)
        S: Photon concentration array (cm^-3)
        N_final: Final carrier concentration (cm^-3)
        S_final: Final photon  concentration (cm^-3)
    """
    
    # Setup integrator ###
    sol = solve_ivp(
        laser_rates,
        t_span=(0, SimConfig.T_END),
        y0=[SimConfig.N_INITIAL, SimConfig.S_INITIAL],
        method='BDF',  # Better for stiff systems
        args=(p, I),
        dense_output=True,
        rtol=1e-7,
        atol=1e-9,
    )
    t_eval = np.linspace(0, SimConfig.T_END, 50000)
    T = sol.t
    N = sol.y[0]
    S = sol.y[1]
    
    return T, N, S, N[-1:], S[-1:]


### Plotting functions ###
def plot_dynamic(t, n, s):
    """Plot dynamic simulation results (carrier and 
            photon concentration vs time).
    
    Args:
        t: Time array (s)
        n: Carrier concentration array (cm^-3)
        s: Photon concentration array (cm^-3)
    """
    
    fig, axarr = plt.subplots(2, sharex=True, figsize=(10, 8))
    
    # Carrier density
    axarr[0].plot(t * 1e9, n, 'g-', linewidth=2)
    axarr[0].set_ylabel("Carrier concentration (cm$^{-3}$)")
    axarr[0].set_title('Laser rate equations - dynamic simulation')
    axarr[0].grid(True, alpha=0.3)
    
    # Photon density
    axarr[1].plot(t * 1e9, s, 'b-', linewidth=2)
    axarr[1].set_ylabel("Photon concentration (cm$^{-3}$)")
    axarr[1].set_xlabel("Time (ns)")
    axarr[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_steady_state(p, s, I):
    """Plot steady-state LI characteristics (power and quantum efficiency
        vs current).
    
    Args:
        p: Laser parameters dictionary
        s: Final photon concentration array (cm^-3)
        I: Current array (A)
    """
    
    # Calculate optical parameters
    f = calculate_mode_frequency(p)
    V = calculate_cavity_volume(p)
    τ_p, τ_α = calculate_photon_lifetime(p)
    
    # Calculate output power (mW)
    P = [constants.h * f * ((i * V) / τ_p) * 1e3 for i in s]
    
    # Calculate quantum efficiency (dimensionless), normalise by current in mA
    QE = [i / j if j > 0 else 0 for i, j in zip(P, I * 1e3)]       
    # Create dual-axis plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Plot power on left axis
    line1 = ax1.plot(I * 1e3, P, 'g-', linewidth=2, label='Output power')
    ax1.set_xlabel('Current (mA)', fontsize=12)
    ax1.set_ylabel('Power (mW)', color='g', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.grid(True, alpha=0.3)
    
    # Plot quantum efficiency on right axis
    line2 = ax2.plot(I * 1e3, QE, 'b-', linewidth=2, label='Quantum efficiency')
    ax2.set_ylabel('Quantum efficiency', color='b', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='b')
    
    plt.title('Steady-state solution: LI characteristics', fontsize=12)
    plt.tight_layout()
    plt.show()
    plt.close()


### Main execution ###
if CALC_MODE == 0:
    # Dynamic simulation
    T, N, S, _, _ = solve(LASER_PARAMS, CURRENT_DYNAMIC)
    plot_dynamic(T, N, S)


elif CALC_MODE == 1:
    # Steady-state LI curve
    S_final = []
    for I in CURRENT_SWEEP:
        _, _, _, _, S_hld = solve(LASER_PARAMS, I)
        S_final.append(S_hld[0])    
    plot_steady_state(LASER_PARAMS, S_final, CURRENT_SWEEP)
