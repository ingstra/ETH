#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mp
from qutip import *
from numpy import pi
from numpy.linalg import matrix_power as mpow

# energy levels
N = 3
id = qeye(N) # identity matrix


# basis states
g = fock_dm(N,0)
e = fock_dm(N,1)
f = fock_dm(N,2)


# qubit frequencies in MHz
w1_ge = 5.708390 * 1000
w2_ge = 5.202204 * 1000
w3_ge = 4.877051 * 1000
w4_ge = 4.383388 * 1000

w1_ef = 5.447309 * 1000
w2_ef = 4.927032 * 1000
w3_ef = 4.599720 * 1000
w4_ef = 4.096882 * 1000


# self-Kerr coefficients (anharmonicities) in MHz
alpha_1 = -261.081457
alpha_2 = -275.172227
alpha_3 = -277.331082
alpha_4 = -286.505059

# nearest-neighbor couplings in MHz
J_12 = 3.8 * 2*pi
J_23 = 3.4 * 2*pi
J_34 = 3.5 * 2*pi

# decay times T1 in µs
q1_T1 = 25.111199
q2_T1 = 17.987260
q3_T1 = 25.461490
q4_T1 = 46.371173


# destruction operators
a1 = tensor(destroy(N), id, id, id)
a2 = tensor(id, destroy(N), id, id)
a3 = tensor(id, id, destroy(N), id)
a4 = tensor(id, id, id, destroy(N))

a = [a1, a2, a3, a4]

# drive frequency
wd = w1_ge
# detunings
delta1_ge = w1_ge - wd
delta2_ge = w2_ge - wd
delta3_ge = w3_ge - wd
delta4_ge = w4_ge - wd

delta1_ef = w1_ef - wd
delta2_ef = w2_ef - wd
delta3_ef = w3_ef - wd
delta4_ef = w4_ef - wd


tlist = np.linspace(0,10,200)

# Hamiltonian in frame rotating with drive freq wd
H_qubit2 = delta1_ge*tensor(e,id,id,id) + delta1_ef*tensor(f,id,id,id)

H_qubit = delta1_ge*a1.dag()*a1 + alpha_1*a1.dag()**2*a1**2 + delta2_ge*a2.dag()*a2 + alpha_2*a2.dag()**2*a2**2 + delta3_ge*a3.dag()*a3 + alpha_3*a3.dag()**2*a3**2 + delta4_ge*a4.dag()*a4 + alpha_4*a4.dag()**2*a4**2



H_selfKerr = alpha_1*a1.dag()**2*a1**2 + alpha_2*a2.dag()**2*a2**2 + alpha_3*a3.dag()**2*a3**2 + alpha_4*a4.dag()**2*a4**2
H_coupling = J_12*(a1.dag()*a2 + a2.dag()*a1) + J_23*(a2.dag()*a3 + a3.dag()*a2) + J_34*(a3.dag()*a4 + a4.dag()*a3)


H = H_qubit + H_coupling + H_selfKerr

def H_drive_coeff(t, args):
    sigma2 = .1
    tg=6 # gate time
    pulse_tmax = 10
    A = pi/2
    B = 1
    if t <= pulse_tmax:
        retval =  A*np.exp(-(t-0.5*tg)**2/(2*sigma2))/np.sqrt(2*pi*sigma2)
    else:
        retval = 0
    return retval

pulse = [H_drive_coeff(t,'') for t in tlist]

plt.plot(tlist,pulse,'.',label='pulse')

H_drive = a1 + a1.dag()

H_tot = [H, [H_drive, H_drive_coeff]]





# solve master equation


# initial state vacuum
vac = basis(N,0)
psi0 = tensor(vac, vac, vac, vac)


e_ops=[]

# collapse operators
c_ops = []


opt = Options(nsteps=30000, atol=1e-10, rtol=1e-8)
result = mesolve(H_tot, psi0, tlist, c_ops, e_ops, options=opt)

rho = result.states

nf = f*f.dag()
ne = e*e.dag()
ng = g*g.dag()

exp0000 = expect(tensor(ng,id,id,id), rho)
exp1000 = expect(tensor(ne,id,id,id), rho)
exp2000 = expect(tensor(nf,id,id,id), rho)
expx1xx = expect(tensor(id,ne,id,id), rho)
expxx1x = expect(tensor(id, id, ne, id), rho)
expxxx1 = expect(tensor(id, id, id, ne), rho)
expx2xx = expect(tensor(id,nf,id,id), rho)
expxx2x = expect(tensor(id, id, nf, id), rho)
expxxx2 = expect(tensor(id, id, id, nf), rho)

#plt.plot(tlist, exp0000,label='g')
plt.plot(tlist, exp1000,label='1xxx')
plt.plot(tlist, exp2000,label='2xxx')

plt.plot(tlist, expx1xx, label='x1xx')
plt.plot(tlist, expxx1x, label='xx1x')
plt.plot(tlist, expxxx1, label='xxx1')
plt.plot(tlist, expx2xx, label='x2xx')
plt.plot(tlist, expxx2x, label='xx2x')
plt.plot(tlist, expxxx2, label='xxx2')

plt.legend()
plt.show()



# https://books.google.se/books?id=5MGSDwAAQBAJ&pg=PA54&lpg=PA54&dq=kerr+term+anharmonicity&source=bl&ots=rF5T7oyYDX&sig=ACfU3U39hhawzbSvp2f1USznXmPDV3qyfA&hl=en&sa=X&ved=2ahUKEwjC5PXRpeflAhXQI1AKHQ83CCQQ6AEwDXoECAgQAQ#v=onepage&q=kerr%20term%20anharmonicity&f=false

# anharmonicity of qubit: anharmonicity
# anharmonicity of cavity: self-kerr
# dispersive shift: cross-kerr
