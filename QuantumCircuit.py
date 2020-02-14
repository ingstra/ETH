import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

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



class QuantumCircuit:

    def destroy(self, n):
        """
        Function that constructs the destruction operator for the n:th qubit
        """
        # Creation Operator
        a = qt.destroy(3)
        # Identity
        eye = qt.qeye(3)
        return qt.tensor([a if n == j else eye for j in range(self.num_q)])


    def __init__(self, qubits, cpl_map = None, cpl = None):
        """
        init
        """
        self.qubits = qubits
        self.num_q = len(qubits) # number of qubits
        self.cpl_map = cpl_map # coupling map
        self.cpl = cpl # couplings

        # Rotating frame
        rotating_freq = 0#2*np.pi * 5.0

        # Construct the Qubit Hamiltonian
        H_qubits = 0
        for idx,elem in enumerate(self.qubits):
            omega = elem.resonance_freq
            alpha = elem.anharmonicity

            # Creation operator for the i:th qubit
            c = self.destroy(idx)
            H_qubits += (omega - rotating_freq)*c.dag()*c  + (alpha/2) * c.dag()**2 * c**2

        # Construct the Interaction/Coupling Hamiltonian
        H_cpl = 0
        for i in range(len(self.cpl)):
            x, y = self.cpl_map[i]
            g = self.cpl[i]
            a_x = self.destroy(x)
            a_y = self.destroy(y)
            H_cpl += g*(a_x.dag()*a_y + a_y.dag()*a_x)

        # Create the total Hamiltonian
        self.Hamiltonian = H_qubits + H_cpl


    def Rx(self, n, theta):
        """
        X rotation
        n: is the qubit
        theta: is the angle
        """

        # hamiltonian in rotating frame of drive
        # drive at qubit resonance freq
        drive_freq = self.qubits[n].resonance_freq
        print(drive_freq)
        rotating_freq = drive_freq
        a = qt.destroy(3)
        H_qubit = (self.qubits[n].resonance_freq - rotating_freq)*a.dag()*a + (self.qubits[n].anharmonicity/2) * a.dag()**2 * a**2

        # Drive Hamiltonian (since we have drag pulses, we need both x and y drive)
        #
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
        drive_x = H_drive_coeff_x(tlist, args)
        # out of-phase component
        drive_y = H_drive_coeff_y(tlist, args)
        # Integrate along the given time-axis using the composite trapezoidal rule
        B = theta/(2*np.abs(np.trapz(drive_x + 1j*drive_y, x=tlist)))
        print('Amplitude =',B)

        # Master equation simulation
        args = {'amp': B, 'qscale': 0.032, 'freq': (rotating_freq - drive_freq), 'L': L, 'sigma': sigma}
        result = qt.mesolve(H, self.qubits[n].state, tlist, [], [], args=args)
        final_state = result.states[-1] # final state

        # unitary matrix to into transmon frame
        rotate_U = qt.Qobj([[1,0,0],
                   [0,np.exp(1j*(self.qubits[n].resonance_freq - rotating_freq)*t_total),0],
                   [0,0,np.exp(1j*(self.qubits[n].resonance_freq + self.qubits[n].anharmonicity - rotating_freq)*t_total)]])

        # rotated transmon state
        final_state = rotate_U*final_state
        self.qubits[n].state = final_state



class Qubit:
    def __init__(self, resonance_freq, anharmonicity, T1=0, T2=0):
        self.resonance_freq = resonance_freq
        self.anharmonicity = anharmonicity
        self.T1 = T1
        self.T2 = T2
        self.state = qt.fock_dm(3, 0) # 3-level qubit initialized in ground state
