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
import time
from scipy.optimize import curve_fit
###############################################################################
# Dashboard
###############################################################################
# Settings ###
SS = 0              # 0 is dynamic, 1 is steadt-state LI, 2 is Spectrum
MM = 1              # 0 single mode, 1 multiple modes
COAT = 0            # Turn coating on or off


# Inputs values ###
Im = 6                  # Pumping current (mA)
Is = 20                 # LI stop
nI = 40                 # LI no points
n_eff = 4.2             # Effective refractive index
tn = 2.7                # Carrier relaxation time (ns)
al_c = 1                # Cavity loss (cm^-1)
lam_0 = 1550            # Add chosen centre WL for calc (nm)
cl = 250                # Cavity len (um)
cw = 2                  # Cavity width (um)
ct = 8                  # Cavity thickness (nm)
N_tr = 1.8e18           # Transparency conc (cm^-3)
Gam = 0.032             # Confinement factor Coldren
A = 1/tn                # Carrier recomb (s^1)
B = 0.8e-10             # Bimolecular recomb (cm^3s^-1)
C = 3.5e-30             # Auger recom (cm^6s^-1)
a = 5.34e-16            # Gain slope eff (cm^2)
eps = 1.5e-17           # Gain compression factor (cm^3)
M = 15                  # Gain band FWHM (no of modes)
G_m = 45.6              # Enter threshold mo gain single mode (cm^-1)
B_0 = 0.869e-4          # Spontaneous emission factor

# Time contraints and initial conditions ###
t1 = 2.5e-9                       # Time to run calc for (s)
dt = 1e-14                      # Time step for calc (s)
t0 = 0                          # Time start for calc (s)
S0 = 0                          # Initial photon conc (cm^-3)
N0 = 1e16                       # Initial carrier conc (cm^-3)

# Physical contants ###
q = 1.60217663e-19              # Electron charge (C)
h = 6.62607004e-34              # Plank's contant (Js)
c = 2.99792458e8                # SOL (ms^-1)


###############################################################################
# Pre-calculations
###############################################################################
# Convertions ###
lam_0 = lam_0/1e9
tn = tn/1e9
cl = cl/1e4
cw = cw/1e4
ct = ct/1e7

# Current edit ###
if SS == 0 or 2:
    I = [Im/1e3]
if SS == 1:     #LI
    Im = np.linspace(0, Is, nI)          # Generate multiple I for LI curve (mA)
    I = [Im[i]/1e3 for i in range(len(Im))]

# Calc device volume (cm^3) ###
V_a = cl*ct*cw
print(V_a)

# Group vel (cms^-1) & Freq (Hz) calc ###
f = c/lam_0
v_g = ((c*100)/n_eff)
wl_c = lam_0

# Single-mode operation values ###
if MM == 0:
    G_m = np.array([G_m])
    m0 = 0                          # Manually set m0 for single mode simulation

# Read in modal gain ###
if MM == 1:
    Input = np.load('Inv_Gain.npz')
    WL_m = Input['WL']
   # G_m = Input['G_m']
    # minus 1 main modes
    G_m = [32.39612396486444, 32.40326493867929, 32.417450348732444, 32.41163230324251, 32.429554693370356, 32.421999581465606, 32.43112856600654, 32.417980409726894, 32.42423593226248, 32.41296427709051, 32.41603725232717, 32.421930679971524, 32.42997194712663, 32.4349247902813, 32.46158539270661, 32.50140762524224, 32.59292549088015, 32.721124232352565, 33.24761573806994, 31.46310125216062, 24.197251553199326, 31.46310125216062, 33.24761573806994, 32.721124232352565, 32.59292549088015, 32.50140762524224, 32.46158539270661, 32.4349247902813, 32.42997194712663, 32.421930679971524, 32.41603725232717, 32.41296427709051, 32.42423593226248, 32.417980409726894, 32.43112856600654, 32.421999581465606, 32.429554693370356, 32.41163230324251, 32.417450348732444, 32.40326493867929, 32.39612396486444]

    m0 = int(np.ceil(len(G_m)/2))
    wl_c = WL_m[m0]

# Simualtion input arrays ###
y0 = [N0]                           # Initial conds [N]
T = []                              # Time array output
for i in range(len(G_m)+1):         # Add initial S for each mode
    y0.append(S0)

print("m0 = ", m0)

# Loss calculations ###
if COAT == 0:
    r_l = r_r = (n_eff-1)/(n_eff+1)     # Calc ref from cacity
if COAT == 1:
    r_l = 0.5234                        # Left cavity ref
    r_r = 0.5234                        # Right cavity ref


al_m = (1/cl)*np.log(1/(r_l*r_r))
al = al_c + al_m
tp = 1/(al*v_g)                         # Photon lifetime (s)
print('Unpertubed tp =', tp)
print(v_g)
# Simualtion output arrays ###
P_m = []

###############################################################################
# Main function definition
###############################################################################
def solver(Ii):

    p = [Ii, q, V_a, v_g, tn, Gam]     # Parameters for odes

    # Output arrays
    Y = []                              # Output array
    N = []                              # y[0] Carrier concentration
    S_t = []                            # y[1] Total photon conc in cavity
    S_m = []                            # Photon conc per mode y[1:i]

    # Generate output array for each mode
    for i in range(len(G_m)+2):
        S_m.append([])

    # Setup integrator with desired parameters ###
    if SS == 0 or 2:    # Models oscillations
        r = ode(laser_rates).set_integrator('dopri5', nsteps = 1e6)
    if SS == 1:     # Faster less accurate
        r = ode(laser_rates).set_integrator('vode', method='bdf')
        #r = ode(laser_rates).set_integrator('lsoda', method = 'bdf')
    r.set_f_params(p).set_initial_value(y0, t0)

    # Simulation check ###
    while r.successful() and r.t+dt < t1:
        r.integrate(r.t + dt)
        Y.append(r.y)                   # Makes a list of 1d arrays
        T.append(r.t)

    # Format output ###
    Y = np.array(Y)                     # Convert from list to 2d array
    N = Y[:, 0]
    S_m = Y[:, 2:]
    S_t = [sum(S_m[i]) for i in range(len(S_m))]

    return N, S_t, S_m


# Define equations to be solved ###
def laser_rates(t, y, p):

    # Generate outputs for each mode ###
    dy = np.zeros([len(G_m)+2])
    cur = p[0]

    # Carrier equation ###
    dy[0] = cur/(q*V_a) - Gain(y[0], y[1], 0)*y[1]*v_g - R(y[0])

    # Total carrier conc calc ###
    y[1] = sum([y[i] for i in range(2, len(dy))])    # Total stim emission

    # Calculation for each independent mode ###
    for i in range(len(G_m)):
        dy[i+2] = Gam*Gain(y[0], y[1], i)*y[i+2]*v_g - y[i+2]/tp_m(G_m[i])\
        + Gam*spont(y[0])*B_0


    # Display outputs of each mode ###
    # print(y)

    return dy


###############################################################################
# Supplementary definitions
###############################################################################
# Carrier recomb time tn removed above threshold ###
def R(N):
    x =  A*(N-N_tr) + B*(N-N_tr)**2 + C*(N-N_tr)**3
    y  = "{:e}".format(x)
    #print('Carrier Recomb', y, 's^-1')
    return x

# Spont emssions, approximation from paper ###
def spont(N):
    return (N-N_tr)*B


# Power conversion mW ###
def p_mW(s):
    return h*f*((s*Gam*V_a)/tp)*1e3


# Power conversion dBm ###
def p_dBm(mW):
    return 10*np.log10(mW)


# Gain log calc p277 C&C ###
def Gain(n, s, itr):  # s is modal total for photon conc
    x =  (a*n)*np.log(n/N_tr)*(1/(1+abs(itr-m0)/M**2))*(1/(1+eps*s))
    y  = "{:e}".format(x)
    #print('Gain Factor', y,'s')
    return x

# tp_m used to factor in varying loss for each mode ###
def tp_m(G):

    x = 1/(v_g*G - al_c*v_g)
    y  = "{:e}".format(x)
    #print('Photon lifetime', y,'s')
    return x

#Define your function
def func(x, a, b, c):
    return a*np.exp(-b*x) + c


# Fitting for relaxation oscillation freq
def omega_f(s):

    s0 = [s[i][m0-1] for i in range(len(T))]
    start = np.argmax(s0)
    stop = np.argmax(T)
    fit_s = s0[start:stop]
    fit_t = T[start:stop]
    print(fit_s[0])
    #popt, pcov = curve_fit(func, fit_t, fit_s, p0=[fit_s[0], 1e-9, fit_t[-1]])
    #popt, pcov = curve_fit(func, fit_t, fit_s)
    scale_x= np.average(fit_t)
    scale_y= np.average(fit_s)
    popt, pcov = curve_fit(func, fit_t, fit_s, diag=(1./scale_x,1./scale_y))
    fit_plot = [func(i, popt[0],popt[1],popt[2]) for i in fit_t]
    print(popt)
    plt.plot(fit_t, fit_plot, 'r--')
    plt.plot(fit_t, fit_s)
    plt.ylabel("Photon Conc ($cm^{-3}$)")
    plt.title('Relaxation Constant Fitting')
    #plt.ylim(0, 3e16)
    plt.xlabel("Time (s)")
    plt.show()

    return

###############################################################################
# Plotting function definitions
###############################################################################
# Dynamic plotting ###
def plot_dynam_s(s, n):

    f, axarr = plt.subplots(2, sharex=True)     # Two subplots
    axarr[0].plot(T, n, 'g')
    axarr[0].set_ylabel("Carrier Conc ($cm^{-3}$)")
    axarr[0].set_title('Laser-Rate Simulation')
    axarr[1].plot(T, s, 'b')
    axarr[1].set_ylabel("Photon Conc ($cm^{-3}$)")
    axarr[1].set_xlabel("Time (s)")
    plt.show()

    return


# Dynamic plotting ###
def plot_dynam_mm(s):
    plt.plot(T, s)
    plt.ylabel("Photon Conc ($cm^{-3}$)")
    plt.title('Laser-Rate Simulation')
    #plt.ylim(0, 3e16)
    plt.xlabel("Time (s)")
    plt.show()

    return

# Dynamic plotting ###
def plot_dynam_smsr(s):

    s0 = [s[i][m0] for i in range(len(T))]
    s1 = [s[i][m0-1] for i in range(len(T))]
    SMSR = [p_dBm(p_mW(s1[i])) -
            p_dBm(p_mW(s0[i])) for i in range(len(T))]

    plt.plot(T, SMSR)
    plt.ylabel("Supression (dB)")
    plt.title('Dynamic SMSR')
    plt.xlabel("Time (s)")
    plt.show()

    return

# Function for post solver steady-state LI calculations ###
def plot_LI_mm(s):

    # Post solver calculations  Power output (mW)
    P = [p_mW(s[i]) for i in range(len(I))]

    # Plotting MM LI ###
    plt.plot(Im, P)
    plt.xlabel('Current (mA)')
    plt.ylabel('Power (mW)')
    plt.title("Multi-Mode LI")
    plt.show()

    return


# Function for post solver steady-state LI calculations ###
def plot_LI_s(s):

    # Post solver calculations
    P = [p_mW(s[i]) for i in range(len(s))]     # Power output (mW)
    QE = [i/j for i, j in zip(P, Im)]                # Convert to QE

    # Plotting two parameters on one plot ###
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(Im, P, 'g-')
    ax2.plot(Im, QE, 'b-')
    ax1.set_xlabel('Current (mA)')
    ax1.set_ylabel('Power (mW)', color='g')
    ax2.set_ylabel('Quantum Efficiency', color='b')
    plt.title("Steady-State Solution")
    plt.show()

    return


# Function to plot spectrum ###
def plot_spec(s):

    Pow = [p_dBm(p_mW(s[i])) for i in range(len(s))]
    plt.plot(WL_m, Pow)
    plt.xlabel('WL (nm)')
    plt.ylabel('Power (dBm)')
    plt.title('Spectrum')
    plt.show()

    return


###############################################################################
# Main section to call functions
###############################################################################
if __name__ == '__main__':
    tic = time.perf_counter()
    # Dynamic single mode ###
    if SS == 0 and MM == 0:
        P_N, P_St, P_S = solver(I[0])
        toc = time.perf_counter()
        print(f"The code executed in {toc - tic:0.4f} seconds")
        plot_dynam_s(P_S, P_N)

    # Dynamic multi-mode ###
    if SS == 0 and MM == 1:
        P_N, P_St, P_S = solver(I[0])
        toc = time.perf_counter()
        print(f"The code executed in {toc - tic:0.4f} seconds")
        plot_dynam_mm(P_S)
        #omega_f(P_S)
        plot_dynam_smsr(P_S)

    # Steady-state single-mode LI ###
    if SS == 1 and MM == 0:
        for i in range(len(I)):
            P_N, P_St, P_S = solver(I[i])
            P_m.append(P_S[-1])
        toc = time.perf_counter()
        print(f"The code executed in {toc - tic:0.4f} seconds")
        plot_LI_s(P_m)

    # Call to plot LIs ###
    if SS == 1 and MM == 1:
        for i in range(len(I)):
            P_N, P_St, P_S = solver(I[i])
            P_m.append(P_St[-1])
        toc = time.perf_counter()
        print(f"The code executed in {toc - tic:0.4f} seconds")
        plot_LI_mm(P_m)

    # Multimode spectra & LI across WL ###
    if SS == 2 and MM == 1:
        P_N, P_St, P_S = solver(I[0])
        toc = time.perf_counter()
        print(f"The code executed in {toc - tic:0.4f} seconds")
        plot_spec(P_S[-1])

