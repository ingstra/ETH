import numbers
import numpy as np
from collections.abc import Iterable

from qutip.qip.noise import (
    Noise, RelaxationNoise, DecoherenceNoise,
    ControlAmpNoise, RandomNoise, UserNoise, process_noise)
from qutip.operators import qutrit_ops
from qutip.qip.pulse import Pulse

__all__ = ['ETHNoise']

class ETHNoise(UserNoise):
    """
    The decoherence on each qubit characterized by two time scales t1 and t2.

    Parameters
    ----------
    t1: float or list, optional
        Characterize the decoherence of amplitude damping for
        each qubit.
    t2: float or list, optional
        Characterize the decoherence of dephasing for
        each qubit.
    targets: int or list, optional
        The indices of qubits that are acted on. Default is on all
        qubits

    Attributes
    ----------
    t1: float or list
        Characterize the decoherence of amplitude damping for
        each qubit.
    t2: float or list
        Characterize the decoherence of dephasing for
        each qubit.
    targets: int or list
        The indices of qubits that are acted on.
    """
    def __init__(self, t1=None, t2=None, targets=None):
        self.t1 = t1
        self.t2 = t2
        self.targets = targets

    def _T_to_list(self, T, N):
        """
        Check if the relaxation time is valid

        Parameters
        ----------
        T: list of float
            The relaxation time
        N: int
            The number of component systems.

        Returns
        -------
        T: list
            The relaxation time in Python list form
        """
        if (isinstance(T, numbers.Real) and T > 0) or T is None:
            return [T] * N
        elif isinstance(T, Iterable) and len(T) == N:
            if all([isinstance(t, numbers.Real) and t > 0 for t in T]):
                return T
        else:
            raise ValueError(
                "Invalid relaxation time T={},"
                "either the length is not equal to the number of qubits, "
                "or T is not a positive number.".format(T))

    def get_noisy_dynamics(self, pulses, dims):
        """
        Return a list of Pulse object with only trivial ideal pulse (H=0) but
        non-trivial relaxation noise.

        Parameters
        ----------
        dims: list, optional
            The dimension of the components system, the default value is
            [3,3...,3] for qutrit system.

        Returns
        -------
        lindblad_noise: list of :class:`qutip.qip.Pulse`
            A list of Pulse object with only trivial ideal pulse (H=0) but
            non-trivial relaxation noise.
        """
        if isinstance(dims, list):
            for d in dims:
                if d != 3:
                    raise ValueError(
                        "Relaxation noise is defined only for qutrit system")
            N = len(dims)
        else:
            N = dims

        self.t1 = self._T_to_list(self.t1, N)
        self.t2 = self._T_to_list(self.t2, N)
        if len(self.t1) != N or len(self.t2) != N:
            raise ValueError(
                "Length of t1 or t2 does not match N, "
                "len(t1)={}, len(t2)={}".format(
                    len(self.t1), len(self.t2)))

        lindblad_noise = Pulse(None, None)
        [sig11, sig22, sig33, sig12, sig23, sig31] = qutrit_ops()

        if self.targets is None:
            targets = range(N)
        else:
            targets = self.targets
        for qu_ind in targets:
            t1 = self.t1[qu_ind]
            t2 = self.t2[qu_ind]
            if t1 is not None:
                op = 1/np.sqrt(t1) * sig12
                lindblad_noise.add_lindblad_noise(op, qu_ind, coeff=True)
            if t2 is not None:
                # Keep the total dephasing ~ exp(-t/t2)
                if t1 is not None:
                    if 2*t1 < t2:
                        raise ValueError(
                            "t1={}, t2={} does not fulfill "
                            "2*t1>t2".format(t1, t2))
                    T2_eff = 1./(1./t2-1./(2.*t1))
                else:
                    T2_eff = t2
                op = 1/np.sqrt(T2_eff) * sig22
                lindblad_noise.add_lindblad_noise(op, qu_ind, coeff=True)
        return lindblad_noise
