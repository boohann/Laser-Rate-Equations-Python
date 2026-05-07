##########################################################
### Program to simulate laser rate equations in Python ###
############### Niall Boohan 2018 ########################
############### boohann@tcd.ie ###########################
##########################################################


### Theory and equations sourced from:
# Title: Extraction of DFB laser rate equation parameters for system simulation purposes
# Authors: J. C. Cartledge and R. C. Srinivasan
# DOI: 10.1109/50.580827
# URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=580827&isnumber=12618

# Title: Extraction of laser rate equations parameters for representative simulations
#        of metropolitan-area transmission systems and network
# Authors: I. Tomkos, I. Roudas, R. Hesse, N. Antoniades, A. Boskovic, R. Vodhanel
# DOI: https://doi.org/10.1016/S0030-4018(01)01230-5
# URL: https://www.sciencedirect.com/science/article/abs/pii/S0030401801012305?via%3Dihub

### Import necessary libraries ###
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

### Select calculation ###
CALC = 0                                            # 0 is dynamic, 1 is steady-state LI
current_single = 20/1e3                             # Set constant current for dynamic calculation (A)      
current_sweep = np.linspace(0, 50, 100)             # Generate multiple I for steady-state LI curve (mA)
current_sweep = [i/1e3 for i in current_sweep]      # Multiple I (A)
          


### Simulation input parameters ###
LASER_PARAMS = {
    'α': 5,              # Lasing mode cavity loss (cm^-1)
    'n': 3.2,            # Cavity refractive index
    'L': 400,            # Cavity length (um)
    'w': 2,              # Cavity width  (um)
    'h_active': 100,     # Height active region (nm)
    'λ':1300,            # Lasing mode wavelength (nm)
    'r_l': 0.5,          # Left facet amplitude reflectivity
    'r_r': 0.5,          # Right facet amplitude reflectivity
    'β':1e-4,            # Spontaneous Emission Factor
    'Γ':0.15,            # Quantum well confinement factor
    'τ_n':1.0e-9,        # Carrier relaxation time in seconds (s)
    'g':1.5e-5,          # Gain slope constant (cm^3s^-1)
    'N_tr':1e17,         # Transparency carrier density (cm^-3)
    'ε':1.5e-17,         # Gain compression factor (cm^3)
}

# Config hard-coded values ###
class SimConfig:
    T_END = 5e-9
    DT = 1e-13
    N_INITIAL = 1e16
    S_INITIAL = 0

### Define equations to be solved ###

def laser_rates(t, y, p, I):       
  dy = np.zeros([2])
  τ_p = 1/((constants.c/(p['L']*1e-6))*np.log(1/(p['r_l']*p['r_r'])))        # Photon round-trip time in cavity (s)
  τ_α = 1/(constants.c*p['α']*100)                                           # Photon lifetime from cavity loss (s)
  V = p['L']*p['w']*p['h_active']*(1e-15)                                    # Volume active region  (cm^3) 
  
  # dN/dt = I/(q*V) - N/τ_n - dg*S(N-N_tr)/(1+ε*S)
  # dS/dt = Γ*g_0*dg*S*(N-N_tr)/(1+ε*S) - S/τ_p - Γ*β*N/τ_n
  dy[0] = (I/(constants.q*V)) - (y[0]/p['τ_n']) -  p['g']*(y[0]-p['N_tr'])*(y[1]/(1+p['ε']*y[1]))
  dy[1] = p['Γ']*p['g']*(y[0]-p['N_tr'])*(y[1]/(1+p['ε']* y[1])) - y[1]/(p['τ_p']+p['τ_α']) + (p['Γ']*p['β']*y[0])/p['τ_n']     
  return dy
        

def solve(p, I):

    Y=[]; T=[]     # Create empty output lists, N=Y[:, 0],  N=Y[:, 1]
 
    ### Setup integrator with desired parameters ###
    # Runge-Kutta must be used as a solver, minimum 4th order
    r = ode(laser_rates).set_integrator('dopri5', nsteps = 1e4)
    r.set_f_params([p, I]).set_initial_value([SimConfig.N_INITIAL, SimConfig.S_INITIAL], 0)

    
    ### Simulation check ###
    while r.successful() and r.t+SimConfig.DT < SimConfig.T_END:
        r.integrate(r.t + SimConfig.DT)
        Y.append(r.y)        # Makes a list of 1d arrays
        T.append(r.t)
    

    ### Format output ###
    Y = np.array(Y)          # Convert from list to 2d array

    ### Take final value for steady-state LI ###
    return T, Y[:, 0], Y[:, 1], Y[:, 0][-1:], Y[:, 1][-1:]


### Dynamic plotting ###
def plot_dynamic(t, n, s):

    f, axarr = plt.subplots(2, sharex=True) # Two subplots, the axes array is 1-d
    axarr[0].plot(t, n, 'g-')
    axarr[0].set_ylabel("Carrier conc ($cm^{-3}$)")
    axarr[0].set_title('Laser-rate simulation')
    axarr[1].plot(t, s, 'b-')
    axarr[1].set_ylabel("Photon concentration ($cm^{-3}$)")
    axarr[1].set_xlabel("Time (s)")
    plt.show()
    plt.close()

    


### Function for post-solver steady-state LI calculations and plotting ###
def plot_steady_state(p, s, I):
    f = constants.c/(p['λ']/1e9)                                     # Frequency of mode calculate (Hz)   
    P = [constants.h*p['f']*((i*p['V'])/p['τ_p'])*1e3 for i in s]    # Power output (mW)
    QE = [i/j for i,j in zip(P, I)]                                  # Convert for quantum efficiency

    ### Plotting two parameters on one plot ###
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(I, P,  'g-')
    ax2.plot(I, QE, 'b-')
    ax1.set_xlabel('Current (mA)')
    ax1.set_ylabel('Power (mW)', color='g')
    ax2.set_ylabel('Quantum efficiency', color='b')
    plt.title("Steady-state solution")
    plt.show()
    plt.close()

    


### Dynamic ###
if(CALC == 0):
    T, N, S, _, _ = solve(LASER_PARAMS, current_single)
    plot_dynamic(T, N, S)


### Steady-state ###
if(CALC == 1):
    S_final = []
    for i in current_sweep:
        _, _, _, _, S_hld = solve(LASER_PARAMS, i)
        S_final.append(S_hld)
    plot_steady_state(S_final, LASER_PARAMS, current_sweep)
