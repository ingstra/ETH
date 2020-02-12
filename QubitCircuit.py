import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

class QubitCircuit:
    def __init__(self, nbrOfQubits=4, FockDim=3):
        self.nbrOfQubits = nbrOfQubits
        self.FockDim = FockDim

        self.qubits = []
        for i in range(nbrOfQubits):
            self.qubits.append(qt.fock_dm(FockDim, 0))

    def testfunc(self):
        a = qt.destroy(self.FockDim)
        self.qubits[0]= a.dag()*self.qubits[0]*a



def H_drive_coeff_x(t, args):

    # Amplitude, needs to be optimized to get perfect pi pulse or pi/2 pulse
    B = args['amp']

    # DRAG-parameter, needs to be optimized to get no phase errors in a pi/2 pulse
    q = args['qscale']
    drive_freq = args['freq']
    L = args['L']
    sigma = args['sigma']

    # E(t)
    E = B * np.exp(-pow(t-0.5*L, 2) / (2*pow(sigma,2)))

    I = E
    Q = q * (t-0.5*L) / sigma * E # dE/dt
    return I*np.cos(drive_freq * t) + Q*np.sin(drive_freq * t)

def H_drive_coeff_y(t, args):

    # Amplitude, needs to be optimized to get perfect pi pulse or pi/2 pulse
    B = args['amp']
    # DRAG-parameter, needs to be optimized to get no phase errors in a pi/2 pulse
    q = args['qscale']
    drive_freq = args['freq']
    L = args['L']
    sigma = args['sigma']

    # E(t)
    E = B * np.exp(-pow(t-0.5*L, 2) / (2*pow(sigma,2)))

    I = E
    Q = -q * (t-0.5*L) / sigma * E # dE/dt
    return Q*np.cos(drive_freq * t) + I*np.sin(drive_freq * t)






class Qubit:
    def __init__(self, resonance_freq, anharmonicity, fock_dim=3):
        self.fock_dim = fock_dim
        self.resonance_freq = resonance_freq
        self.anharmonicity = anharmonicity

        self.state = qt.fock_dm(fock_dim, 0)



    def Rx(self, theta, drive_freq):
        # hamiltonian in rotating frame of drive
        # drive at qubit resonance freq
        rotating_freq = drive_freq
        a = qt.destroy(self.fock_dim)
        H_qubit = (self.resonance_freq - rotating_freq)*a.dag()*a + (self.anharmonicity/2) * a.dag()**2 * a**2

        # Drive Hamiltonian (since we have drag pulses, we need both x and y drive)
        H_drive_x = a.dag() + a
        H_drive_y = 1j*(a.dag() - a)

        # Gaussian width in μs
        sigma = 0.010 # = 10 ns

        # Gate time in μs
        L = 0.050 # = 50 ns

        # total Hamiltonian
        H = [H_qubit,
             [H_drive_x,H_drive_coeff_x],
             [H_drive_y,H_drive_coeff_y]
        ]

        # Total time in μs
        t_total = L
        tlist = np.linspace(0,t_total,500)

        args = {'amp': 1, 'qscale': 0.018, 'freq': (rotating_freq - drive_freq), 'L': L, 'sigma': sigma}

        # in-phase component
        I = H_drive_coeff_x(tlist, args)
        # out of-phase component
        Q = H_drive_coeff_y(tlist, args)

        # Integrate along the given time-axis using the composite trapezoidal rule
        B = theta/(2*np.abs(np.trapz(I + 1j*Q, x=tlist)))
        print('Amplitude =',B)

        # Master equation simulation
        args = {'amp': B, 'qscale': 0.032, 'freq': (rotating_freq - drive_freq), 'L': L, 'sigma': sigma}
        result = qt.mesolve(H, self.state, tlist, [], [], args=args)
        rho = result.states
        final_state = rho[-1] # final state

        # unitary matrix to into transmon frame
        rotate_U = qt.Qobj([[1,0,0],
                   [0,np.exp(1j*(self.resonance_freq - rotating_freq)*t_total),0],
                   [0,0,np.exp(1j*(self.resonance_freq + self.anharmonicity - rotating_freq)*t_total)]])

        # rotated transmon state
        final_state = rotate_U*final_state
        self.state = final_state





# Qubit frequency (0->1) in MHz
omega = 2*np.pi*5.708390 * 1000

# Self-Kerr coefficient (anharmonicity) in MHz
alpha = - 2*np.pi*261.081457



N=3

q1 = Qubit(omega, alpha, N)

theta=np.pi/3

print(q1.state)

q1.Rx(theta, omega)

print(q1.state)

# project onto qubit subspace
#qubit_state = (basis(2,0)*basis(3,0).dag() + basis(2,1)*basis(3,1).dag())*final_state
qubit_state = qt.Qobj(q1.state[:2,:2])
print(qubit_state)

# Create a Bloch sphere
b = qt.Bloch()
b.add_states(qubit_state)
b.show()
plt.show()
