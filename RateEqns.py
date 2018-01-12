### Program to simulate laser rate equations ###
import numpy as np
#import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

### Universal Constants ###
q = 1.6e-19                            # Electron charge (C)
c = 3.0e8                               # Speed of light (ms^-1)

### Initial conditions ###
I = 2.0e-2                                  # 50mA pumping current
L = 1.e-3                                  # Device length (m)
W = 2.0e-6                                    # Device width (m)
D = 1e-7                                  # Active region thickness (m)
tn = 4.0e-9                                   # Carrier relaxation time in seconds (s)
tp = 1.0e-12                                 # Photon lifetime in cavity (s)
Nth = 1.5e24                               # Threshold carrier density (m^-3)
EPS = 1.25e-23                                     # Gain compression factor (m^3)
Beta = 1.0e-5                                 # Spontaneous Emission Factor
Alpha = 1.53e3                                 # Gain (m^-1)

B = 1.0e-16                                   # Bimolecular recombination factor (m^3s^-1)
C = 3.0e-41                                   # Auger recombination factor (m^6s^-1)

Gamma = 0.17                               # Confinement factor
indeff = 3.1857                              # Effective refractive index

V = L*W*D
s0 = 0.0
n0 = 0.0

tmax = 1e-3                                 # Simulation length time
t = np.linspace(0, tmax, num = 100)     # Time vector
#nt = len(t)                                 # Total number of time steps

#n = N/Nth
#s = S/Sth
Ith = (Nth*q*V)/tn
vg = c/indeff
Vst = vg*Gamma*Alpha*V
Sth = 1/(Vst*tn) 
tau_norm = tn/tp


def rates(Y, t, I, Ith, B, tn, Nth, C, EPS, Sth, Beta, tau_norm, full_output=1):
    """
    S = Photon number in simulation
    N = Carrier number in simulation
    """
    n, s = Y

    dYdt = [I/Ith - B*tn*Nth*(n**2) - C*tn*(Nth**2)*(n**3) - ((n - 1)*s)/(1 - EPS*Sth*s), Nth/(Sth*(n - 1)*s) + (Beta*B*tn*(Nth**2))/(Sth*(n**2)) - tau_norm*s]
    return dYdt

Y0 = n0, s0           # Defines the initial condition for carrier density and photon density in the cavity




### Solve ODE ###
#backend = 'vode'
#backend = 'dopri5'
#backend = 'dop853'
#odeint(rates, Y0, t, args = (I, Ith, B, tn, Nth, C, EPS, Sth, Beta, tau_norm)).set_integrator('vode', method = 'bdf', order = 15)
#set_integrator('vode', method = 'bdf', order = 15)
sol = odeint(rates, Y0, t, args = (I, Ith, B, tn, Nth, C, EPS, Sth, Beta, tau_norm))


'''
n = sol[:, 1]
s = sol[:, 0]          


S = s*Sth
N = n*Nth

### Plot results ###
fig = plt.figure()

ax1 = fig.add_subplot(211)
ax1.plot(t, N, 'g')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('N', color='g')

ax2 = fig.add_subplot(212)
ax2.plot(t, S, 'b')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('S', color='b')


plt.show()
'''
