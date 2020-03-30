import numpy as np

from qutip.qip.circuit import QubitCircuit, Gate
from qutip.qip.compiler.gatecompiler import GateCompiler


__all__ = ['ETHCompiler']


class ETHCompiler(GateCompiler):
    """
    Decompose a :class:`qutip.QubitCircuit` into
    the pulse sequence for the processor.

    Parameters
    ----------
    N: int
        The number of qubits in the system.

    params: dict
        A Python dictionary contains the name and the value of the parameters,
        such as laser frequency, detuning etc.

    global_phase: bool
        Record of the global phase change and will be returned.

    num_ops: int
        Number of Hamiltonians in the processor.

    Attributes
    ----------
    N: int
        The number of the component systems.

    params: dict
        A Python dictionary contains the name and the value of the parameters,
        such as laser frequency, detuning etc.

    num_ops: int
        Number of control Hamiltonians in the processor.

    gate_decomps: dict
        The Python dictionary in the form of {gate_name: decompose_function}.
        It saves the decomposition scheme for each gate.

    global_phase: bool
        Record of the global phase change and will be returned.
    """
    def __init__(self, N, params, global_phase, num_ops):
        super(ETHCompiler, self).__init__(
            N=N, params=params, num_ops=num_ops)
        self.gate_decomps = {"RZ": self.rz_dec,
                             "RX": self.rx_dec,
                             "GLOBALPHASE": self.globalphase_dec
                             }
        self.N = N
        self.global_phase = global_phase

    def decompose(self, gates):
        # TODO further improvement can be made here,
        # e.g. merge single qubit rotation gate, combine XX gates etc.
        self.dt_list = []
        self.coeff_list = []
        for gate in gates:
            if gate.name not in self.gate_decomps:
                raise ValueError("Unsupported gate %s" % gate.name)
            self.gate_decomps[gate.name](gate)

        tlist = self.dt_list
        coeffs = self.coeff_list

        #tlist, coeffs = super(ETHCompiler, self).decompose(gates)
        return tlist, coeffs, self.global_phase

    def rz_dec(self, gate):
        """
        Compiler for the RZ gate
        """
        # todo

    def rx_dec(self, gate):
        """
        Compiler for the RX gate
        """
        # Gaussian width in μs
        sigma = 0.010 # = 10 ns

        # Gate time in μs
        L = 0.050 # = 50 ns

        def H_drive_coeff_x(t, args):

            # Amplitude, needs to be optimized to get perfect pi pulse or pi/2 pulse
            B = args['amp']

            # DRAG-parameter, needs to be optimized to get no phase errors in a pi/2 pulse
            q = args['qscale']

            drive_freq = args['freq']

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

            # E(t)
            E = B * np.exp(-pow(t-0.5*L, 2) / (2*pow(sigma,2)))

            I = E
            Q = -q * (t-0.5*L) / sigma * E # dE/dt
            return Q*np.cos(drive_freq * t) + I*np.sin(drive_freq * t)

        # Total time in μs
        t_total = L
        tlist = np.linspace(0,t_total,500)

        amp = 63.45380720748712 * gate.arg_value / np.pi

        args = {'amp': amp, 'qscale': 0.032, 'freq': 0}

        # in-phase component
        I = H_drive_coeff_x(tlist, args)
        # out of-phase component
        Q = H_drive_coeff_y(tlist, args)

        self.dt_list.append(tlist)
        self.coeff_list.append(I)
        self.coeff_list.append(Q)

    def globalphase_dec(self, gate):
        """
        Compiler for the GLOBALPHASE gate
        """
        self.global_phase += gate.arg_value
