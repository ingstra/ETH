import qutip as q
import numpy as np
import matplotlib.pyplot as plt

class QuantumCircuit:
    def __init__(self, qubits, num_lvl = 3, cpl_map = None, cpl = None):
        self.qubits = qubits
        self.num_q = len(qubits) # number of qubits
        self.num_lvl = num_lvl # number of qubit levels
        self.cpl_map = cpl_map # coupling map
        self.cpl = cpl # couplings

        # Rotating frame
        rotating_freq = 0#2*np.pi * 5.0

        # Creation Operator
        a = q.destroy(self.num_lvl)

        # Identity
        eye = q.qeye(self.num_lvl)

        # Function that constructs the creation operator for the n:th qubit
        def create(n):
            return q.tensor([a if n == j else eye for j in range(self.num_q)])

        # Construct the Qubit Hamiltonian
        H_qubits = 0
        for idx,elem in enumerate(self.qubits):
            omega = elem.resonance_freq
            alpha = elem.anharmonicity

            # Creation operator for the i:th qubit
            c = create(idx)

            H_qubits += (omega - rotating_freq)*c.dag()*c  + (alpha/2) * c.dag()**2 * c**2

        # Construct the Interaction/Coupling Hamiltonian
        H_cpl = 0
        for i in range(len(self.cpl)):
            x, y = self.cpl_map[i]
            g = self.cpl[i]
            a_x = create(x)
            a_y = create(y)
            H_cpl += g*(a_x.dag()*a_y + a_y.dag()*a_x)

        # Create the total Hamiltonian
        self.Hamiltonian = H_qubits + H_cpl

    def rx(self, n, theta):
        """
        n: is the qubit
        theta: is the angle
        """

        # hamiltonian in rotating frame of drive
        # drive at qubit resonance freq
        drive_freq = self.qubits[n].resonance_freq

        # Drive Hamiltonian
        # (since we have drag pulses, we need both x and y drive)
           c = create(n)
        H_drive_x = c.dag() + c
        H_drive_y = 1j*(c.dag() - c)

        # Gaussian width in (ns)
        sigma = 10

        # Gate time in (ns)
        L = 50

        # total Hamiltonian
        H = [H_qubits,
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

class Qubit:
    def __init__(self, resonance_freq, anharmonicity, T1=0, T2=0):
        self.resonance_freq = resonance_freq
        self.anharmonicity = anharmonicity
        self.T1 = T1
        self.T2 = T2
