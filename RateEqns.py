##########################################################
### Program to simulate laser-rate equations in Python ###
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

### Mode select ###
Mode = 1        # 0 is dynamic, 1 is steadt-state LI

### Simualtion Outputs ###
N     = []      # y[0] Carrier concentration
S     = []      # y[1] Photon concentration
T     = []      # Time array output
N_end = []      # Take final N value for steady-state behaviour
S_end = []      # Take final S value for steady-state behaviour

### Simualtion input  parameters ###
IA      = 20                                      # Pumping current (mA)
I       = IA/1e3                                  # Pumping current (A)
iIA     = np.linspace(0, 50, 100)                  # Generate multiple I for LI curve (mA)
iI      = [x/1e3 for x in iIA]                    # Multiple I (A)
q       = 1.6e-19                                 # Electron charge (C)
V       = 2e-11                                   # Device volume (cm^3)
tn      = 1.0e-9                                  # Carrier relaxation time in seconds (s)
g0      = 1.5e-5                                  # Gain slope constant (cm^3s^-1)
Nth     = 1e18                                    # Threshold carrier density (cm^-3)
EPS     = 1.5e-17                                 # Gain compression factor (cm^3)
Gamma   = 0.2                                     # Confinement factor
tp      = 1.0e-12                                 # Photon lifetime in cavity (s)
Beta    = 1.0e-4                                  # Spontaneous Emission Factor
h       = 6.62607004e-34                          # Plank's contant (Js)
c       = 2.99792458e8                            # SOL (ms^-1)
WL      = 1300                                    # WL (nm)
f       = c/(WL/1e9)                              # Frequency (Hz)

def call_solv(x):

    ### Ensures global values of S, N and T are updated from this function ###
    global S
    global N
    global T
    
    ### Define equations to be solved ###
    def laser_rates(t, y, p):
        
        dy = np.zeros([2])
        dy[0] = (x/(q* V)) - (y[0]/tn) -  g0*(y[0] - Nth)*(y[1]/(1 + EPS* y[1]))
        dy[1] = Gamma* g0* (y[0] - Nth)*(y[1]/(1 + EPS* y[1])) - y[1]/tp + (Gamma* Beta* y[0]) / tn
        
        return dy
        

    ### Time, initial conditions & add paramters ###  
    t0 = 0; tEnd = 1e-8; dt = 1e-13                     # Time constraints
    y0 = [1e16, 0]                                      # Initial conditions [N, S]
    Y=[]; T=[]                                          # Create empty lists
    p = [I, q, V, tn, g0, Nth, EPS, Gamma, tp, Beta]    # Parameters for odes


    ### Setup integrator with desired parameters ###
    r = ode(laser_rates).set_integrator('vode', method = 'bdf')
    #r = ode(Laser_rates).set_integrator('dopri5', nsteps = 1e4)
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


### Dynamic plotting ###
def plot_dynam():

    f, axarr = plt.subplots(2, sharex=True) # Two subplots, the axes array is 1-d
    axarr[0].plot(T, N, 'G')
    axarr[0].set_ylabel("Carrier Conc ($cm^{-3}$)")
    axarr[0].set_title('Laser-Rate Simulation')
    axarr[1].plot(T, S, 'B')
    axarr[1].set_ylabel("Photon Conc ($cm^{-3}$)")
    axarr[1].set_xlabel("Time (s)")
    plt.show()

    return;


### Function for post solver steady-state LI calculations ###
def plot_SS():
    
    ### Post solver calculations
    P          = [h*f*((i*V)/tp)*1e3 for i in S_end]        # Power output (mW)
    QE         = [i/j for i,j in zip(P, iIA)]               # Convert for quantum efficiency

    ### Plotting two parameters on one plot ###
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(iIA, P,  'g-')
    ax2.plot(iIA, QE, 'b-')
    ax1.set_xlabel('Current (mA)')
    ax1.set_ylabel('Power (mW)', color='g')
    ax2.set_ylabel('Quantum Efficiency', color='b')
    plt.title("Steady-State Solution")
    plt.show()

    return;


### Dynamic mode ###
if(Mode == 0):
    call_solv(I)
    plot_dynam()


### Steady-state mode ###
if(Mode == 1):
    for i in iI:
        call_solv(i)
    plot_SS()
