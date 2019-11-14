#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mp
from qutip import *
from numpy import pi

ustate = basis(3, 0)
excited = basis(3, 1)
ground = basis(3, 2)
N = 2 # Set where to truncate Fock state for cavity

sigma_ge = tensor(qeye(N), ground * excited.dag())  # |g><e|
sigma_ue = tensor(qeye(N), ustate * excited.dag())  # |u><e|
a = tensor(destroy(N), qeye(3))
n_a = tensor(num(N), qeye(3))

c_ops = []
kappa = 1.5 # Cavity decay rate
c_ops.append(np.sqrt(kappa) * a)

gamma = 6  # Atomic decay rate
c_ops.append(np.sqrt(5*gamma/9) * sigma_ue) # Use Rb branching ratio of 5/9 e->u

c_ops.append(np.sqrt(4*gamma/9) * sigma_ge) # 4/9 e->g

# Define time vector
Tend = 15
t = np.linspace(-15, 15, 100)

# Define initial state
psi0 = tensor(basis(N, 0), ustate)

# Define states onto which to project
state_GG = tensor(basis(N, 1), ground)

sigma_GG = state_GG * state_GG.dag()
state_UU = tensor(basis(N, 0), ustate)
sigma_UU = state_UU * state_UU.dag()
g = 5  # coupling strength

# time-independent term
H0 = -g * (sigma_ge.dag() * a + a.dag() * sigma_ge)

# time-dependent term
H1 = (sigma_ue.dag() + sigma_ue)

def H1_coeff(t, args):
    return 25 * np.exp(-(t / 4) ** 2)

H = [H0,[H1,H1_coeff]]

output = mesolve(H, psi0, t, c_ops, [n_a, sigma_UU, sigma_GG])

plt.plot(t, output.expect[0])
#plt.xlim([0,Tend])
plt.show()
