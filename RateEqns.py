'''
###############################################################################
########### Program to simulate laser-rate equations in Python ################
########################## Niall Boohan 2020 ##################################
###################### niall.boohan@tyndall.ie ################################
###############################################################################

BASED ON:
Theory and equations sourced from:
Title: Diode Lasers and Photonic Integrated Circuits
Author: Coldren, Larry A.
year: 1997

NOTES:
-Multimode calculation based on https://en.wikipedia.org/wiki/
 Laser_diode_rate_equations
-Radiative and non-radiative recombination Schubert, E. Fred
'''

# Import necessary libraries ###
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Dashboard
###############################################################################
# Settings ###
SS = 1              # 0 is dynamic, 1 is steadt-state LI
MM = 1              # 0 single mode, 1 multiple modes
LI = 1              # Plot LIs of mm ss
coated = 0          # Turn coating on or off

# Inputs natural units ###
IA = 50                                 # Pumping current (mA)
iIA = np.linspace(0, 50, 10)            # Generate multiple I for LI curve (mA)
cneff = 3.2                             # Effective refractive index
tn = 1                                  # Carrier relaxation time (ns)
al_c = 0                                # Cavity loss (cm^-1)
lam_0 = 1550                            # Add chosen centre WL for calc (nm)
cl = 400                                # Cavity len (um)
cw = 2                                  # Cavity width (um)
ct = 8                                  # Cavity thickness (nm)
S0 = 0                                  # Initial photon conc (cm^-3)
N0 = 1e16                               # Initial carrier conc (cm^-3)
N_tr = 1e18                             # Transparency conc (cm^-3)
Gam = 0.2                               # Confinement factor Coldren
B = 5e-10                               # Bimolecular recomb (cm^3s^-1)
C = 7e-29                               # Auger recom (cm^6s^-1)
time_stop = 1e-9                        # Time to run calc for (s)
time_step = 1e-12                       # Time step for calc (s)
a = 5e-16                               # Gain slope eff (cm^2)
eps = 5e-20                             # Gain compression factor (cm^3)
G_sm = 40                              # Threshold mo gain single mode (cm^-1)
G_sm = np.array([G_sm])

# Physical contants ###
q = 1.60217663e-19                      # Electron charge (C)
h = 6.62607004e-34                      # Plank's contant (Js)
c = 2.99792458e8                        # SOL (ms^-1)

###############################################################################
# Pre-calculations
###############################################################################
# Convert to A, m, s ###
Ip = IA/1e3
iI = [iIA[i]/1e3 for i in range(len(iIA))]
lam_0 = lam_0/1e9
tn = tn/1e9
cl = cl/1e4
cw = cw/1e4
ct = ct/1e7

# Calc device volume (cm^3) ###
V_a = cl*ct*cw


# Group vel (cms^-1) & Freq (Hz) calc ###
f = c/lam_0
v_g = ((c*100)/cneff)

# Simualtion Outputs ###
N = []                      # y[0] Carrier concentration
S = []                      # Photon conc y[i]
T = []                      # Time array output
S_m = []
N_end = []                  # Take final N value for steady-state behaviour
S_end = []                  # Take final S value for steady-state behaviour
S_LI = []
P_m = []

# Read in modal gain ###
if MM == 1:
    Input = np.load('Inv_Gain.npz')
    WL_m = Input['WL']
    G_m = Input['G_m']
    B_0 = 1/len(G_m)                # Mode fact
    m0 = int(np.floor(len(G_m)/2))
    wl_c = WL_m[m0]
    for i in range(len(G_m)):
        S.append([])

m0 = 0
B_0 = 1
M = 10                              # Gain width (a in Inv calc code)
wl_c = lam_0
print("m0 = ", m0)


# Loss calculations ###
if coated == 0:
    r_l = r_r = (cneff-1)/(cneff+1)                   # Calc ref from cacity
if coated == 1:
    r_l = 0.5234                                      # Left cavity ref
    r_r = 0.5234                                      # Right cavity ref


al_m = (1/cl)*np.log(1/(r_l*r_r))
al = al_c + al_m
tp = 1/(al*v_g)                                       # Photon lifetime (s)
print('tp=', tp)

for i, null in enumerate(G_sm):
    print(i)


###############################################################################
# Supplementary definitions
###############################################################################
# Carrier recomb time ###
def R(N):
    return (N-N0)/tn + B*(N-N0)**2 + C*(N-N0)**3


def spont(N):
    return (N-N0)*B


# Power conversion mW ###
def p_mW(S):
    return h*f*((S*Gam*V_a)/tp)*1e3


# Power conversion dBm ###
def p_dBm(mW):
    return 10*np.log10(mW)


# Gain log calc ###
def Gain(tab, ind, G):
    return a*(tab[0])*v_g*np.log((tab[0])/N_tr)\
        *(1/(1+(G[ind]-m0)/M**2))*(1-sum(eps*tab[1:]))


# tp_m used to factor in varying loss for each mode ###
def tp_m(G):
    return v_g*G


###############################################################################
# Main function definition
###############################################################################
def call_solv(cur, g):

    # Ensures global values of S, N and T are updated from this function ###
    global S
    global N
    global T

    # Define equations to be solved ###

    def laser_rates(t, y, p):
        if MM == 0:
            dy = np.zeros([2])
        if MM == 1:
            dy = np.zeros([len(g)+1])
        print(y)
        m_l = sum([Gain(y, i, g)*y[i+1] for i, null in enumerate(g)])
        dy[0] = cur/(q*V_a) - m_l - R(y[0])
        for i, null in enumerate(g):
            dy[i+1] = Gain(y, i, g)*y[i+1] + Gam*spont(y[0])*B_0 \
                - y[i+1]*tp_m(g[i])
        return dy

    # Time, initial conditions & add paramters ###
    t0 = 0
    tEnd = time_stop
    dt = time_step                          # Time constraints
    y0 = [N0]                               # Initial conds [N]
    for i, null in enumerate(g):            # Add initial S for each mode
        y0.append(S0)
    p = [Ip, q, V_a, v_g, tn, Gam, tp]      # Parameters for odes
    Y = []
    T = []                                  # Create empty lists

    # Setup integrator with desired parameters ###
    # r = ode(laser_rates).set_integrator('vode', method='bdf')
    r = ode(laser_rates).set_integrator('dopri5', nsteps=1e4)
    # r = ode(laser_rates).set_integrator('lsoda', method = 'bdf')
    r.set_f_params(p).set_initial_value(y0, t0)

    # Simulation check ###
    while r.successful() and r.t+dt < tEnd:
        r.integrate(r.t + dt)
        Y.append(r.y)        # Makes a list of 1d arrays
        T.append(r.t)

    # Format output ###
    Y = np.array(Y)          # Convert from list to 2d array
    N = Y[:, 0]
    S_t = Y[:, 1]
    S = Y[:, 1:]

    # Take final value for steady-state LI ###
    if SS == 1 and MM == 0:
        S_end.append(S_t[-1])
        N_end.append(N[-1])

        return S_end, N_end

    # Take final value for steady-state LI ###
    if SS == 1 and MM == 1:
        N_end.append(N[-1])
        S_m = S[-1]

        return S_m, N_end

    # Dynamic single mode ###
    if SS == 0 and MM == 0:

        return S_t, N

    # Dynamic multi mode ###
    if SS == 0 and MM == 1:

        return S, N


###############################################################################
# Plotting function definitions
###############################################################################
# Dynamic plotting ###
def plot_dynam():

    f, axarr = plt.subplots(2, sharex=True)     # Two subplots
    axarr[0].plot(T, P_N, 'g')
    axarr[0].set_ylabel("Carrier Conc ($cm^{-3}$)")
    axarr[0].set_title('Laser-Rate Simulation')
    axarr[1].plot(T, P_S, 'b')
    axarr[1].set_ylabel("Photon Conc ($cm^{-3}$)")
    axarr[1].set_xlabel("Time (s)")
    plt.show()

    return


# Dynamic plotting ###
def plot_dynam_mm():

    plt.plot(T, P_S)
    plt.ylabel("Photon Conc ($cm^{-3}$)")
    plt.title('Laser-Rate Simulation')
    plt.ylim(0, 1.8e17)
    plt.xlabel("Time (s)")
    plt.show()

    return


# Function for post solver steady-state LI calculations ###
def plot_LI_mm(S):

    # Post solver calculations  Power output (mW)
    P = [[p_mW(P_m[i][j]) for i in range(len(iI))] for j in range(len(G_m))]

    # Plotting MM LI ###
    for i in range(len(G_m)):
        plt.plot(iIA, P[i])
    plt.xlabel('Current (mA)')
    plt.ylabel('Power (mW)')
    plt.title("Multi-Mode LI")
    plt.show()

    return


# Function for post solver steady-state LI calculations ###
def plot_LI_s(S):
    # Post solver calculations
    P = [p_mW(P_S[i]) for i in range(len(P_S))]     # Power output (mW)
    QE = [i/j for i, j in zip(P, iIA)]              # Convert to QE

    # Plotting two parameters on one plot ###
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(iIA, P, 'g-')
    ax2.plot(iIA, QE, 'b-')
    ax1.set_xlabel('Current (mA)')
    ax1.set_ylabel('Power (mW)', color='g')
    ax2.set_ylabel('Quantum Efficiency', color='b')
    plt.title("Steady-State Solution")
    plt.show()

    return


# Function to plot spectrum ###
def plot_spec():

    Pow = p_dBm(p_mW(P_S))
    plt.plot(WL_m, Pow)
    plt.xlabel('WL (nm)')
    plt.ylabel('Power (dBm)')
    plt.title('Spectrum')
    plt.show()

    return


###############################################################################
# Main section to call functions
###############################################################################
# Dynamic single mode ###
if SS == 0 and MM == 0:
    P_S, P_N = call_solv(Ip, G_sm)
    plot_dynam()

# Dynamic single mode ###
if SS == 0 and MM == 1:
    P_S, P_N = call_solv(Ip, G_m)
    plot_dynam_mm()

# Steady-state single-mode LI ###
if SS == 1 and MM == 0:
    for i in iI:
        P_S, P_N = call_solv(i, G_sm)
    plot_LI_s(S_LI)

# Multimode spectra & LI across WL ###
if SS == 1 and MM == 1:
    P_S, P_N = call_solv(Ip, G_m)
    plot_spec()

if LI == 1:
    for i in iI:
        P_S, P_N = call_solv(i, G_m)
        P_m.append(P_S)

print(P_m)
plot_LI_mm(S_LI)
