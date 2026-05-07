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

### Import necessary libraries ###
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
from scipy import cons

### Select calculation ###
CALC = 0        # 0 is dynamic, 1 is steady-state LI
current_sweep     = np.linspace(0, 50, 100)                 # Generate multiple I for LI curve (mA)
current      = [i/1e3 for i in current_sweep]                    # Multiple I (A)
current = 20/1e3            


### Simulation input parameters ###
LASER_PARAMS = {
    'I': I    # Pumping current (A)
    'α': 5,                 # Cavity lasing mode loss (cm^-1)
    'n': 3.2,               # Cavity refractive index
    'L': 400,                # Cavity length
    'w': 2,                   # Cavity width
    'h_active': 100,            # Height active region (nm)
    'V': L*w*h_active*(1e-15),        # Volume active region  (cm^3)    
    'r_l': 0.5,                # Left amplitude reflectivity
    'r_r': 0.5,             # Left amplitude reflectivity
    'β':1e-4,             # Spontaneous Emission Factor
    'Γ':0.15,            # Quantum well confinement factor
    'τ_n':1.0e-9,                                  # Carrier relaxation time in seconds (s)
    'τ_p':1/((cons.c/(L*1e-6))*np.log(1/(r_l*r_r)))    # Photon round-trip time in cavity (s)
    'τ_α':1/(cons.c*α*100)                             # Photon lifetime material loss (s)
    'g_0':1.5e-5                                  # Gain slope constant (cm^3s^-1)
    'N_tr':1e17                                    # Transparency carrier density (cm^-3)
    'ε':1.5e-17                                 # Gain compression factor (cm^3)
    'λ':1300                                    # WL (nm)
    'f':cons.c/(WL/1e9)                              # Frequency (Hz)   
}

# Config hard-coded values ###
class SimConfig:
    T_START = 0
    T_END = 5e-9
    DT = 1e-13
    N_INITIAL = 1e16
    S_INITIAL = 0

### Define equations to be solved ###
def laser_rates(t, y, p):       
  dy = np.zeros([2])
  dy[0] = (x/(p['q']*p['V'])) - (y[0]/p['τ_n']) -  p['g_0']*(y[0]-p['N_tr'])*(y[1]/(1+p['ε']*y[1]))
  dy[1] = p['Γ']*p['g_0']*(y[0]-p['N_tr'])*(y[1]/(1+p['ε']* y[1])) - y[1]/(p['τ_p']+p['τ_α']) + (p['Γ']*p['β']*y[0])/p['τ_n']     
  return dy
        

def call_solv(p, I):

    Y=[]; T=[]     # Create empty output lists, N=Y[:, 0],  N=Y[:, 1]
 
    ### Setup integrator with desired parameters ###
    # Runge-Kutta must be used as a solver, minimum 4th order
    r = ode(laser_rates).set_integrator('dopri5', nsteps = 1e4)
    r.set_f_params([p, I]).set_initial_value([N_INITIAL, S_INITIAL], T_START)

    
    ### Simulation check ###
    while r.successful() and r.t+DT < T_END:
        r.integrate(r.t + DT)
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
def plot_steady_state(p, s):
    
    ### Post-solver calculations
    P = [cons.h*p['f']*((i*p['V'])/p['tp'])*1e3 for i in s]        # Power output (mW)
    QE = [i/j for i,j in zip(P, p['I'])]              # Convert for quantum efficiency

    ### Plotting two parameters on one plot ###
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(iIA, P,  'g-')
    ax2.plot(iIA, QE, 'b-')
    ax1.set_xlabel('Current (mA)')
    ax1.set_ylabel('Power (mW)', color='g')
    ax2.set_ylabel('Quantum efficiency', color='b')
    plt.title("Steady-state solution")
    plt.show()
    plt.close()

    


### Dynamic ###
if(CALC == 0):
    T, N, S, _, _ = solve(current, LASER_PARAMS)
    plot_dynamic(T, N, S)


### Steady-state ###
if(CALC == 1):
    S_final = []
    for i in current:
        _, _, _, _, S_hld = solve(i, LASER_PARAMS)
        S_final.append(S_hld)
    plot_steady_state(S_final)
