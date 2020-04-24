from collections.abc import Iterable
import warnings
import numbers

import numpy as np
from qutip import *
from qutip.qip.circuit import QubitCircuit
from qutip.qip.device.processor import Processor
from qutip.qip.compiler.gatecompiler import GateCompiler
from qip.ethcompiler import ETHCompiler

__all__ = ['ETHProcessor']

class ETHProcessor(Processor):
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
    correct_global_phase: boolean, optional
        If true, the analytical solution will track the global phase. It
        has no effect on the numerical solution.

    Attributes
    ----------
    params: dict
        A Python dictionary contains the name and the value of the parameters
        in the physical realization, such as laser frequency, detuning etc.
    """

    def __init__(self, correct_global_phase=True):
        self.N = 4 # Number qubits

        self.correct_global_phase = correct_global_phase
        self.spline_kind = "cubic"

        # Qubit frequency in (GHz)
        #self.resonance_freq = [2*np.pi * 5.708390, 2*np.pi * 5.708390, 2*np.pi * 5.708390, 2*np.pi * 5.708390]
        self.resonance_freq = [2*np.pi * 5.708390, 2*np.pi * 5.202204, 2*np.pi * 4.877051, 2*np.pi * 4.383388]

        # Anharmonicity in (GHz)
        #self.anharmonicity = [- 2*np.pi * 0.261081457, - 2*np.pi * 0.261081457, -2*np.pi * 0.261081457, -2*np.pi * 0.261081457]
        # True
        self.anharmonicity = [- 2*np.pi * 0.261081457, - 2*np.pi * 0.275172227, -2*np.pi * 0.277331082, -2*np.pi * 0.286505059]

        self.dims = [3] * self.N  # Three level systems

        self._paras = {}
        self.set_up_params()

    def set_up_params(self):
        """
        Save the parameters in the attribute `params`
        """
        self._paras["Resonance frequency"] = self.resonance_freq
        self._paras["Anharmonicity"] = self.anharmonicity
        #self._paras["Rotating frequency"] = self.rotating_freq

    @property
    def params(self):
        return self._paras

    @params.setter
    def params(self, par):
        self.set_up_params(**par)

    def set_up_ops(self,N):
        """
        Generate the Hamiltonians and save them in the attribute `ctrls`.
        """
        a = destroy(3)
        eye = qeye(3)
        H_qubits = 0
        for m in range(N):
            # Creation operator for the m:th qubit
            b = tensor([a if m == j else eye for j in range(N)])

            # Drive Hamiltonian
            H_drive_x = b.dag() + b
            H_drive_y = 1j*(b.dag() - b)
            self.add_control(H_drive_x, targets = list(range(N)), label="I")
            self.add_control(H_drive_y, targets = list(range(N)), label="Q")

            # Detuning
            self.add_control(b.dag()*b, targets = list(range(N)), label="Detuning")

            # Qubit Hamiltonian
            H_qubits += (self.resonance_freq[m] - self.rotating_freq)*b.dag()*b  + (self.anharmonicity[m]/2) * b.dag()**2 * b**2

        self.add_drift(H_qubits, targets = list(range(N)))
        # Construct the Interaction/Coupling Hamiltonian
        """
        if self.cpl:
            H_cpl = 0
            for i, g in enumerate(self.cpl):
                print(i)
                print(g)
                x, y = self.cpl_map[i]
                b_x = tensor([a if x == j else eye for j in range(N)])
                b_y = tensor([a if y == j else eye for j in range(N)])
                H_cpl += g*(b_x.dag()*b_y + b_y.dag()*b_x)
            self.add_drift(H_qubits+H_cpl, targets = list(range(N)))
        else:
            self.add_drift(H_qubits, targets = list(range(N)))
        """

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
        N = qc.N # Number of qubits in circuit
        dims = [3] * N
        if N > self.N:
            raise TypeError('This processor only supports up to four qubits.')
        super(ETHProcessor, self).__init__(N=N,dims=dims)

        # Setup the rotating frame frequency
        self.rotating_freq = np.mean(self.resonance_freq[0:N])
        self._paras["Rotating frequency"] = self.rotating_freq

        # Create the control and drifts
        self.set_up_ops(N)

        dec = ETHCompiler(
            N, self._paras,
            global_phase=0., num_ops=len(self.ctrls))
        tlist, self.coeffs, self.global_phase = dec.decompose(layers)
        for i in range(len(self.pulses)):
            self.pulses[i].tlist = tlist

        return tlist, self.coeffs

    def run_state(self, init_state=None, analytical=False, qc=None,
                  states=None, **kwargs):
        """
        If `analytical` is False, use :func:`qutip.mesolve` to
        calculate the time of the state evolution
        and return the result. Other arguments of mesolve can be
        given as keyword arguments.
        If `analytical` is True, calculate the propagator
        with matrix exponentiation and return a list of matrices.

        Parameters
        ----------
        init_state: Qobj
            Initial density matrix or state vector (ket).

        analytical: boolean
            If True, calculate the evolution with matrices exponentiation.

        qc: :class:`qutip.qip.QubitCircuit`, optional
            A quantum circuit. If given, it first calls the ``load_circuit``
            and then calculate the evolution.

        states: :class:`qutip.Qobj`, optional
         Old API, same as init_state.

        **kwargs
           Keyword arguments for the qutip solver.

        Returns
        -------
        evo_result: :class:`qutip.Result`
            If ``analytical`` is False,  an instance of the class
            :class:`qutip.Result` will be returned.

            If ``analytical`` is True, a list of matrices representation
            is returned.
        """
        if qc is not None:
            self.load_circuit(qc)
        return super(ETHProcessor, self).run_state(
            init_state=init_state, analytical=analytical,
            states=states, **kwargs)

    def get_ops_labels(self):
        """
        Get the labels for each control Hamiltonian.
        """
        labels = []
        for n in range(self.N):
            labels += ([r"$I_%d$" % n] +
                       [r"$Q_%d$" % n] +
                       [r"$\Delta(t)\omega_%d$" % n])
        return labels

    def get_ops_and_u(self):
        """
        Get the labels for each Hamiltonian.

        Returns
        -------
        ctrls: list
            The list of Hamiltonians
        coeffs: array_like
            The transposed pulse matrix
        """
        return (self.ctrls, self.get_full_coeffs().T)

    def pulse_matrix(self):
        """
        Generates the pulse matrix for the desired physical system.

        Returns
        -------
        t, u, labels:
            Returns the total time and label for every operation.
        """
        #dt = 0.01
        H_ops, H_u = self.get_ops_and_u()
        # FIXME This might becomes a problem if new tlist other than
        # int the default pulses are added.
        tlist = self.get_full_tlist()
        dt = tlist[1] - tlist[0]
        t_tot = tlist[-1]
        n_t = len(tlist)
        n_ops = len(H_ops) # Number of controls = 2

        #t = np.linspace(0, t_tot, n_t) # len(t) = len(tlist)
        t = tlist
        u = H_u.T
        #u = np.zeros((n_ops, n_t))

        return tlist, u, self.get_ops_labels()

    def plot_pulses(self, title=None, noisy=None, figsize=(12, 6), dpi=None):
        """
        Maps the physical interaction between the circuit components for the
        desired physical system.

        Returns
        -------
        fig, ax: Figure
            Maps the physical interaction between the circuit components.
        """
        # TODO add test
        if noisy is not None:
            return super(QuantumDevice, self).plot_pulses(
                title=title, noisy=noisy)
        import matplotlib.pyplot as plt
        t, u, u_labels = self.pulse_matrix()

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        for n, uu in enumerate(u):
            ax.plot(t, u[n], label=u_labels[n])

        ax.axis('tight')
        #ax.set_ylim(-1.5 * 2 * np.pi, 1.5 * 2 * np.pi)
        ax.legend(loc='center left',
                  bbox_to_anchor=(1, 0.5), ncol=(1 + len(u) // 16))
        ax.set_ylabel("Control pulse amplitude (arb. u.)")
        ax.set_xlabel("Time (ns)")
        if title is not None:
            ax.set_title(title)
        fig.tight_layout()
        return fig, ax
