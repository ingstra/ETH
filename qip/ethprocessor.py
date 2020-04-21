import warnings
import numbers

import numpy as np
from qutip import *
from qutip.qip.circuit import QubitCircuit
from qutip.qip.device.processor import Processor
#from qutip.qip.device.modelprocessor import ModelProcessor
from qutip.qip.compiler.gatecompiler import GateCompiler
from qip.ethcompiler import ETHCompiler
from qip.quantumdevice import QuantumDevice

__all__ = ['ETHProcessor']

class ETHProcessor(QuantumDevice):
    """
    The processor based on the physical implementation of
    a ETH processor.
    The available Hamiltonian of the system is predefined.
    For a given pulse amplitude matrix, the processor can
    calculate the state evolution under the given control pulse,
    either analytically or numerically.
    (Only additional attributes are documented here, for others please
    refer to the parent class :class:`qutip.qip.device.ModelProcessor`)

    Parameters
    ----------
    N: int
        The number of qubits in the system.

    resonance_freq: list or float
        The frequency of the qubits

    anharmonicity: list or float
        The anharmonicity of the qubits

    # NOT IN USE
    cpl_map: list
        cpl_map[[i, j], [k, l]]: i-th (k-th) and j-th (l-th) qubits are coupled

    # NOT IN USE
    cpl: list
        cpl_list[i]: coupling strength of i-th couple (defined by cpl_map)

    # NOT IN USE
    t1: list or float
        Characterize the decoherence of amplitude damping for
        each qubit. A list of size ``N`` or a float for all qubits.

    # NOT IN USE
    t2: list of float
        Characterize the decoherence of dephasing for
        each qubit. A list of size ``N`` or a float for all qubits.
    """

    def __init__(self, N,
                 resonance_freq,
                 anharmonicity,
                 cpl_map = None,
                 cpl = None,
                 correct_global_phase=True, t1=None, t2=None):
        super(ETHProcessor, self).__init__(
        N, correct_global_phase=correct_global_phase, t1=t1, t2=t2)

        self.correct_global_phase = correct_global_phase
        self.spline_kind = "cubic"
        self.resonance_freq = resonance_freq
        self.anharmonicity = anharmonicity
        self.cpl_map = cpl_map
        self.cpl = cpl
        self.dims = [3] * N  # Three level systems

        self._paras = {}
        self.set_up_params(
        N=N,resonance_freq=resonance_freq,anharmonicity=anharmonicity)
        self.set_up_ops(N)

    def set_up_ops(self, N):
        """
        Generate the Hamiltonians and save them in the attribute `ctrls`.

        Parameters
        ----------
        N: int
            The number of qubits in the system.
        """
        # Annihilation operator for the transmon
        a = destroy(3)
        eye = qeye(3)

        """
        for m in range(N):
            H_qubit = (self.resonance_freq[m] - self.rotating_freq)*a.dag()*a+(self.anharmonicity[m]/2)*pow(a.dag(),2)*pow(a,2)
            self.add_drift(H_qubit, targets = m)
        """

        H_qubits = 0
        for m in range(N):
            # Creation operator for the m:th qubit
            b = tensor([a if m == j else eye for j in range(N)])

            # Drive Hamiltonian (since we have drag pulses, we need both x and y drive)
            H_drive_x = b.dag() + b
            H_drive_y = 1j*(b.dag() - b)

            self.add_control(H_drive_x, targets = list(range(N)), label="I")
            self.add_control(H_drive_y, targets = list(range(N)), label="Q")

            H_qubits += (self.resonance_freq[m] - self.rotating_freq)*b.dag()*b  + (self.anharmonicity[m]/2) * b.dag()**2 * b**2

        self.add_drift(H_qubits, targets = list(range(N)))

    def set_up_params(self, N, resonance_freq, anharmonicity):
        """
        Save the parameters in the attribute `params` and check the validity.

        Parameters
        ----------
        N: int
            The number of qubits in the system.

        resonance_freq: list or float
            The frequency of the qubits

        anharmonicity: list or float
            The anharmonicity of the qubits

        rotating_freq: float
            The rotating frame frequency
        """
        if isinstance(self.resonance_freq, numbers.Real):
            self.resonance_freq = [self.resonance_freq] * N
        self._paras["resonance_freq"] = self.resonance_freq

        if isinstance(self.anharmonicity, numbers.Real):
            self.anharmonicity = [self.anharmonicity] * N
        self._paras["anharmonicity"] = self.anharmonicity

        self.rotating_freq = np.mean(self.resonance_freq)

        self._paras["rotating_freq"] = self.rotating_freq

    def get_ops_labels(self):
        """
        Get the labels for each control Hamiltonian.
        """
        return ([r"$a^\dagger_%d + a_%d$" % (n,n) for n in range(self.N)] +
                [r"$i(a^\dagger_%d - a_%d)$" % (n,n) for n in range(self.N)])

    def load_circuit(self, qc):
        """
        Decompose a :class:`qutip.QubitCircuit` in to the control
        amplitude generating the corresponding evolution.

        Parameters
        ----------
        qc: :class:`qutip.QubitCircuit`
            Takes the quantum circuit to be implemented.

        Returns
        -------
        tlist: array_like
            A NumPy array specifies the time of each coefficient

        coeffs: array_like
            A 2d NumPy array of the shape (len(ctrls), len(tlist)). Each
            row corresponds to the control pulse sequence for
            one Hamiltonian.
        """
        gates = qc.gates
        layers = qc.layers()
        dec = ETHCompiler(
            self.N, self._paras,
            global_phase=0., num_ops=len(self.ctrls))
        tlist, self.coeffs, self.global_phase = dec.decompose(layers)
        for i in range(len(self.pulses)):
            self.pulses[i].tlist = tlist

        return tlist, self.coeffs

    def optimize_circuit(self, qc):
        """
        Take a quantum circuit/algorithm and convert it into the
        optimal form/basis for the desired physical system.

        Parameters
        ----------
        qc: :class:`qutip.QubitCircuit`
            Takes the quantum circuit to be implemented.

        Returns
        -------
        qc: :class:`qutip.QubitCircuit`
            The circuit representation with elementary gates
            that can be implemented in this model.
        """
        # THIS HAVEN'T BEEN WORKED ON
        self.qc0 = qc
        self.qc1 = self.qc0.resolve_gates(
            basis=["ISWAP", "CSIGN", "RX", "RZ"])
        return self.qc1
