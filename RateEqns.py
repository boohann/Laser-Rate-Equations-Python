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

### Select calculation ###
CALC = 0        # 0 is dynamic, 1 is steady-state LI

### Simulation Outputs ###
N     = []      # y[0] Carrier concentration
S     = []      # y[1] Photon concentration
T     = []      # Time array output
N_end = []      # Take the final N value for steady-state behaviour
S_end = []      # Take the final S value for steady-state behaviour

### Simulation input parameters ###
IA      = 20                                      # Pumping current (mA)
I       = IA/1e3                                  # Pumping current (A)
iIA     = np.linspace(0, 50, 100)                 # Generate multiple I for LI curve (mA)
iI      = [x/1e3 for x in iIA]                    # Multiple I (A)
α       = 5                                       # Cavity lasing mode loss (cm^-1)
n       = 3.2                                     # Cavity refractive index
q       = 1.6e-19                                 # Electron charge (C)
L       = 400                                     # Cavity length (um)
w       = 2                                       # Cavity width (um)
h       = 100                                     # Height of active region (nm)
V       = L*w*h*(1e-15)                           # Device volume (cm^3), cm^3 conversion factor
r_l     = 0.5                                     # Left facet reflectivity
r_r     = 0.5                                     # Right facet reflectivity
tn      = 1.0e-9                                  # Carrier relaxation time in seconds (s)
g0      = 1.5e-5                                  # Gain slope constant (cm^3s^-1)
Nth     = 1e18                                    # Threshold carrier density (cm^-3)
EPS     = 1.5e-17                                 # Gain compression factor (cm^3)
Gamma   = 0.15                                    # Confinement factor
Beta    = 1.0e-4                                  # Spontaneous Emission Factor
h       = 6.62607004e-34                          # Plank's constant (Js)
c       = 2.99792458e8                            # SOL (ms^-1)
WL      = 1300                                    # WL (nm)
f       = c/(WL/1e9)                              # Frequency (Hz)
tp      = 1/((c/(L*1e-6))*np.log(1/(r_l*r_r)))    # Photon round-trip time in cavity (s)
tα      = 1/(c*α*100)                             # Photon lifetime material loss (s)



def call_solv(x):

    ### Ensures global values of S, N and T are updated from this function ###
    global S
    global N
    global T
    
    ### Define equations to be solved ###
    def laser_rates(t, y, p):
        
        dy = np.zeros([2])
        dy[0] = (x/(q* V)) - (y[0]/tn) -  g0*(y[0] - Nth)*(y[1]/(1 + EPS* y[1]))
        dy[1] = Gamma* g0* (y[0] - Nth)*(y[1]/(1 + EPS* y[1])) - y[1]/(tp+tα) + (Gamma* Beta* y[0]) / tn
        
        return dy
        

    ### Time, initial conditions & add paramters ###  
    t0 = 0; tEnd = 5e-9; dt = 1e-13                     # Time constraints
    y0 = [1e16, 0]                                      # Initial conditions [N, S]
    Y=[]; T=[]                                          # Create empty lists
    p = [I, q, V, tn, g0, Nth, EPS, Gamma, tp, Beta]    # Parameters for odes


    ### Setup integrator with desired parameters ###
    # Runge-Kutta must be used as a solver, minimum 4th order
    r = ode(laser_rates).set_integrator('dopri5', nsteps = 1e4)
    r.set_f_params(p).set_initial_value(y0, t0)

    
    ### Simualtion check ###
    while r.successful() and r.t+dt < tEnd:
        r.integrate(r.t + dt)
        Y.append(r.y)        # Makes a list of 1d arrays
        T.append(r.t)
    

    ### Format output ###
    Y = np.array(Y)          # Convert from list to 2d array
    N = Y[:, 0] 
    S = Y[:, 1] 

    ### Take final value for steady-state LI ###
    S_end.append(S[-1:])
    N_end.append(N[-1:])

    return;


### Dynamic plotting ###
def plot_dynam():

    f, axarr = plt.subplots(2, sharex=True) # Two subplots, the axes array is 1-d
    axarr[0].plot(T, N, 'g-')
    axarr[0].set_ylabel("Carrier conc ($cm^{-3}$)")
    axarr[0].set_title('Laser-rate simulation')
    axarr[1].plot(T, S, 'b-')
    axarr[1].set_ylabel("Photon concentration ($cm^{-3}$)")
    axarr[1].set_xlabel("Time (s)")
    plt.show()

    return;


### Function for post-solver steady-state LI calculations and plotting ###
def plot_SS():
    
    ### Post-solver calculations
    P = [h*f*((i*V)/tp)*1e3 for i in S_end]        # Power output (mW)
    QE = [i/j for i,j in zip(P, iIA)]              # Convert for quantum efficiency

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

    return;


### Dynamic mode ###
if(CALC == 0):
    call_solv(I)
    plot_dynam()


### Steady-state mode ###
if(CALC == 1):
    for i in iI:
        call_solv(i)
    plot_SS()
