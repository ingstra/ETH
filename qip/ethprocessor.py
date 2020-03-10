import numpy as np
from qutip.operators import sigmax, sigmay, sigmaz, identity
from qutip.tensor import tensor
from qutip.qip.circuit import QubitCircuit
from qutip.qip.device.processor import Processor
from qutip.qip.device.modelprocessor import ModelProcessor, GateDecomposer

class ETHProcessor(ModelProcessor):
    """
    The processor based on the physical implementation of
    the ETH quantum circuit processor.

    Parameters
    ----------
    N: int
        The number of qubits in the system.

    correct_global_phase: float
        Save the global phase, the analytical solution
        will track the global phase.
        It has no effect on the numerical solution.

    sx: int or list
        The delta for each of the qubits in the system.

    sz: int or list
        The epsilon for each of the qubits in the system.

    sxsy: int or list
        The interaction strength for each of the qubit pair in the system.

    t1: list or float
        Characterize the decoherence of amplitude damping for
        each qubit. A list of size ``N`` or a float for all qubits.

    t2: list of float
        Characterize the decoherence of dephasing for
        each qubit. A list of size ``N`` or a float for all qubits.

    Attributes
    ----------
    N: int
        The number of component systems.

    ctrls: list
        A list of the control Hamiltonians driving the evolution.

    tlist: array_like
        A NumPy array specifies the time of each coefficient.

    coeffs: array_like
        A 2d NumPy array of the shape, the length is dependent on the
        spline type

    t1: list
        Characterize the decoherence of amplitude damping for
        each qubit.

    t2: list
        Characterize the decoherence of dephasing for
        each qubit.

    noise: :class:`qutip.qip.Noise`, optional
        A list of noise objects. They will be processed when creating the
        noisy :class:`qutip.QobjEvo` from the processor or run the simulation.

    dims: list
        The dimension of each component system.
        Default is dim=[2,2,2,...,2]

    spline_kind: str
        Type of the coefficient interpolation.
        Note that they have different requirement for the length of ``coeffs``.

        -"step_func":
        The coefficient will be treated as a step function.
        E.g. ``tlist=[0,1,2]`` and ``coeffs=[3,2]``, means that the coefficient
        is 3 in t=[0,1) and 2 in t=[2,3). It requires
        ``coeffs.shape[1]=len(tlist)-1`` or ``coeffs.shape[1]=len(tlist)``, but
        in the second case the last element has no effect.

        -"cubic": Use cubic interpolation for the coefficient. It requires
        ``coeffs.shape[1]=len(tlist)``

    sx: list
        The delta for each of the qubits in the system.

    sz: list
        The epsilon for each of the qubits in the system.

    sxsy: list
        The interaction strength for each of the qubit pair in the system.

    sx_ops: list
        A list of sigmax Hamiltonians for each qubit.

    sz_ops: list
        A list of sigmaz Hamiltonians for each qubit.

    sxsy_ops: list
        A list of tensor(sigmax, sigmay)
        interacting Hamiltonians for each qubit.

    sx_u: array_like
        Pulse matrix for sigmax Hamiltonians.

    sz_u: array_like
        Pulse matrix for sigmaz Hamiltonians.

    sxsy_u: array_like
        Pulse matrix for tensor(sigmax, sigmay) interacting Hamiltonians.
    """
    def __init__(self, N, correct_global_phase, t1, t2):
        super(SpinChain, self).__init__(
            N, correct_global_phase=correct_global_phase, t1=t1, t2=t2)
        self.correct_global_phase = correct_global_phase
        self.spline_kind = "step_func"
        # params and ops are set in the submethods

    def set_up_ops(self, N):
        """
        Generate the Hamiltonians for the spinchain model and save them in the
        attribute `ctrls`.

        Parameters
        ----------
        N: int
            The number of qubits in the system.
        """
        # sx_ops
        self.ctrls += [tensor([sigmax() if m == n else identity(2)
                               for n in range(N)])
                       for m in range(N)]
        # sz_ops
        self.ctrls += [tensor([sigmaz() if m == n else identity(2)
                               for n in range(N)])
                       for m in range(N)]
        # sxsy_ops
        for n in range(N - 1):
            x = [identity(2)] * N
            x[n] = x[n + 1] = sigmax()
            y = [identity(2)] * N
            y[n] = y[n + 1] = sigmay()
            self.ctrls.append(tensor(x) + tensor(y))

    def set_up_params(self, sx, sz):
        """
        Save the parameters in the attribute `params` and check the validity.

        Parameters
        ----------
        sx: float or list
            The coefficient of sigmax in the model

        sz: flaot or list
            The coefficient of sigmaz in the model

        Notes
        -----
        The coefficient of sxsy is defined in the submethods.
        All parameters will be multiplied by 2*pi for simplicity
        """
        sx_para = super(SpinChain, self)._para_list(sx, self.N)
        self._paras["sx"] = sx_para
        sz_para = super(SpinChain, self)._para_list(sz, self.N)
        self._paras["sz"] = sz_para

    @property
    def sx_ops(self):
        return self.ctrls[: self.N]

    @property
    def sz_ops(self):
        return self.ctrls[self.N: 2*self.N]

    @property
    def sxsy_ops(self):
        return self.ctrls[2*self.N:]

    @property
    def sx_u(self):
        return self.coeffs[: self.N]

    @property
    def sz_u(self):
        return self.coeffs[self.N: 2*self.N]

    @property
    def sxsy_u(self):
        return self.coeffs[2*self.N:]

    def load_circuit(self, qc, setup):
        """
        Decompose a :class:`qutip.QubitCircuit` in to the control
        amplitude generating the corresponding evolution.

        Parameters
        ----------
        qc: :class:`qutip.QubitCircuit`
            Takes the quantum circuit to be implemented.

        setup: string
            "linear" or "circular" for two sub-calsses.

        Returns
        -------
        tlist: array_like
            A NumPy array specifies the time of each coefficient

        coeffs: array_like
            A 2d NumPy array of the shape (len(ctrls), len(tlist)). Each
            row corresponds to the control pulse sequence for
            one Hamiltonian.
        """
        gates = self.optimize_circuit(qc).gates

        dec = SpinChainGateDecomposer(
            self.N, self._paras, setup=setup,
            global_phase=0., num_ops=len(self.ctrls))
        self.tlist, self.coeffs, self.global_phase = dec.decompose(gates)

        return self.tlist, self.coeffs

    def adjacent_gates(self, qc, setup="linear"):
        """
        Method to resolve 2 qubit gates with non-adjacent control/s or target/s
        in terms of gates with adjacent interactions for linear/circular spin
        chain system.

        Parameters
        ----------
        qc: :class:`qutip.QubitCircuit`
            The circular spin chain circuit to be resolved

        setup: Boolean
            Linear of Circular spin chain setup

        Returns
        -------
        qc: :class:`qutip.QubitCircuit`
            Returns QubitCircuit of resolved gates for the qubit circuit in the
            desired basis.
        """
        qc_t = QubitCircuit(qc.N, qc.reverse_states)
        swap_gates = ["SWAP", "ISWAP", "SQRTISWAP", "SQRTSWAP", "BERKELEY",
                      "SWAPalpha"]
        N = qc.N

        for gate in qc.gates:
            if gate.name == "CNOT" or gate.name == "CSIGN":
                start = min([gate.targets[0], gate.controls[0]])
                end = max([gate.targets[0], gate.controls[0]])

                if (setup == "linear" or
                        (setup == "circular" and (end - start) <= N // 2)):
                    i = start
                    while i < end:
                        if (start + end - i - i == 1 and
                                (end - start + 1) % 2 == 0):
                            # Apply required gate if control and target are
                            # adjacent to each other, provided |control-target|
                            # is even.
                            if end == gate.controls[0]:
                                qc_t.add_gate(gate.name, targets=[i],
                                              controls=[i + 1])
                            else:
                                qc_t.add_gate(gate.name, targets=[i + 1],
                                              controls=[i])

                        elif (start + end - i - i == 2 and
                              (end - start + 1) % 2 == 1):
                            # Apply a swap between i and its adjacent gate,
                            # then the required gate if and then another swap
                            # if control and target have one qubit between
                            # them, provided |control-target| is odd.
                            qc_t.add_gate("SWAP", targets=[i, i + 1])
                            if end == gate.controls[0]:
                                qc_t.add_gate(gate.name, targets=[i + 1],
                                              controls=[i + 2])
                            else:
                                qc_t.add_gate(gate.name, targets=[i + 2],
                                              controls=[i + 1])
                            qc_t.add_gate("SWAP", [i, i + 1])
                            i += 1

                        else:
                            # Swap the target/s and/or control with their
                            # adjacent qubit to bring them closer.
                            qc_t.add_gate("SWAP", [i, i + 1])
                            qc_t.add_gate("SWAP", [start + end - i - 1,
                                                   start + end - i])
                        i += 1

                elif (end - start) < N - 1:
                    """
                    If the resolving has to go backwards, the path is first
                    mapped to a separate circuit and then copied back to the
                    original circuit.
                    """

                    temp = QubitCircuit(N - end + start)
                    i = 0
                    while i < (N - end + start):

                        if (N + start - end - i - i == 1 and
                                (N - end + start + 1) % 2 == 0):
                            if end == gate.controls[0]:
                                temp.add_gate(gate.name, targets=[i],
                                              controls=[i + 1])
                            else:
                                temp.add_gate(gate.name, targets=[i + 1],
                                              controls=[i])

                        elif (N + start - end - i - i == 2 and
                              (N - end + start + 1) % 2 == 1):
                            temp.add_gate("SWAP", targets=[i, i + 1])
                            if end == gate.controls[0]:
                                temp.add_gate(gate.name, targets=[i + 2],
                                              controls=[i + 1])
                            else:
                                temp.add_gate(gate.name, targets=[i + 1],
                                              controls=[i + 2])
                            temp.add_gate("SWAP", [i, i + 1])
                            i += 1

                        else:
                            temp.add_gate("SWAP", [i, i + 1])
                            temp.add_gate("SWAP",
                                          [N + start - end - i - 1,
                                           N + start - end - i])
                        i += 1

                    j = 0
                    for gate in temp.gates:
                        if (j < N - end - 2):
                            if gate.name in ["CNOT", "CSIGN"]:
                                qc_t.add_gate(gate.name, end + gate.targets[0],
                                              end + gate.controls[0])
                            else:
                                qc_t.add_gate(gate.name,
                                              [end + gate.targets[0],
                                               end + gate.targets[1]])
                        elif (j == N - end - 2):
                            if gate.name in ["CNOT", "CSIGN"]:
                                qc_t.add_gate(gate.name, end + gate.targets[0],
                                              (end + gate.controls[0]) % N)
                            else:
                                qc_t.add_gate(gate.name,
                                              [end + gate.targets[0],
                                               (end + gate.targets[1]) % N])
                        else:
                            if gate.name in ["CNOT", "CSIGN"]:
                                qc_t.add_gate(gate.name,
                                              (end + gate.targets[0]) % N,
                                              (end + gate.controls[0]) % N)
                            else:
                                qc_t.add_gate(gate.name,
                                              [(end + gate.targets[0]) % N,
                                               (end + gate.targets[1]) % N])
                        j = j + 1

                elif (end - start) == N - 1:
                    qc_t.add_gate(gate.name, gate.targets, gate.controls)

            elif gate.name in swap_gates:
                start = min([gate.targets[0], gate.targets[1]])
                end = max([gate.targets[0], gate.targets[1]])

                if (setup == "linear" or
                        (setup == "circular" and (end - start) <= N // 2)):
                    i = start
                    while i < end:
                        if (start + end - i - i == 1 and
                                (end - start + 1) % 2 == 0):
                            qc_t.add_gate(gate.name, [i, i + 1])
                        elif ((start + end - i - i) == 2 and
                              (end - start + 1) % 2 == 1):
                            qc_t.add_gate("SWAP", [i, i + 1])
                            qc_t.add_gate(gate.name, [i + 1, i + 2])
                            qc_t.add_gate("SWAP", [i, i + 1])
                            i += 1
                        else:
                            qc_t.add_gate("SWAP", [i, i + 1])
                            qc_t.add_gate("SWAP", [start + end - i - 1,
                                                   start + end - i])
                        i += 1

                else:
                    temp = QubitCircuit(N - end + start)
                    i = 0
                    while i < (N - end + start):

                        if (N + start - end - i - i == 1 and
                                (N - end + start + 1) % 2 == 0):
                            temp.add_gate(gate.name, [i, i + 1])

                        elif (N + start - end - i - i == 2 and
                              (N - end + start + 1) % 2 == 1):
                            temp.add_gate("SWAP", [i, i + 1])
                            temp.add_gate(gate.name, [i + 1, i + 2])
                            temp.add_gate("SWAP", [i, i + 1])
                            i += 1

                        else:
                            temp.add_gate("SWAP", [i, i + 1])
                            temp.add_gate("SWAP", [N + start - end - i - 1,
                                                   N + start - end - i])
                        i += 1

                    j = 0
                    for gate in temp.gates:
                        if(j < N - end - 2):
                            qc_t.add_gate(gate.name, [end + gate.targets[0],
                                                      end + gate.targets[1]])
                        elif(j == N - end - 2):
                            qc_t.add_gate(gate.name,
                                          [end + gate.targets[0],
                                           (end + gate.targets[1]) % N])
                        else:
                            qc_t.add_gate(gate.name,
                                          [(end + gate.targets[0]) % N,
                                           (end + gate.targets[1]) % N])
                        j = j + 1

            else:
                qc_t.add_gate(gate.name, gate.targets, gate.controls,
                              gate.arg_value, gate.arg_label)

        return qc_t

    def eliminate_auxillary_modes(self, U):
        return U

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
        self.qc0 = qc
        self.qc1 = self.adjacent_gates(self.qc0)
        self.qc2 = self.qc1.resolve_gates(
            basis=["SQRTISWAP", "ISWAP", "RX", "RZ"])
        return self.qc2


class LinearSpinChain(SpinChain):
    """
    A processor based on the physical implementation of
    a linear spin chain qubits system.
    The available Hamiltonian of the system is predefined.
    The processor can simulate the evolution under the given
    control pulses either numerically or analytically.

    Parameters
    ----------
    N: int
        The number of qubits in the system.

    correct_global_phase: float
        Save the global phase, the analytical solution
        will track the global phase.
        It has no effect on the numerical solution.

    sx: int or list
        The delta for each of the qubits in the system.

    sz: int or list
        The epsilon for each of the qubits in the system.

    sxsy: int or list
        The interaction strength for each of the qubit pair in the system.

    t1: list or float, optional
        Characterize the decoherence of amplitude damping for
        each qubit.

    t2: list of float, optional
        Characterize the decoherence of dephasing for
        each qubit.
    """
    def __init__(self, N, correct_global_phase=True,
                 sx=0.25, sz=1.0, sxsy=0.1, t1=None, t2=None):

        super(LinearSpinChain, self).__init__(
            N, correct_global_phase=correct_global_phase,
            sx=sx, sz=sz, sxsy=sxsy, t1=t1, t2=t2)
        self.set_up_params(sx=sx, sz=sz, sxsy=sxsy)
        self.set_up_ops(N)

    def set_up_ops(self, N):
        super(LinearSpinChain, self).set_up_ops(N)

    def set_up_params(self, sx, sz, sxsy):
        # Doc same as in the parent class
        super(LinearSpinChain, self).set_up_params(sx, sz)
        sxsy_para = self._para_list(sxsy, self.N-1)
        self._paras["sxsy"] = sxsy_para

    @property
    def sxsy_ops(self):
        return self.ctrls[2*self.N: 3*self.N-1]

    @property
    def sxsy_u(self):
        return self.coeffs[2*self.N: 3*self.N-1]

    def load_circuit(self, qc):
        return super(LinearSpinChain, self).load_circuit(qc, "linear")

    def get_ops_labels(self):
        return ([r"$\sigma_x^%d$" % n for n in range(self.N)] +
                [r"$\sigma_z^%d$" % n for n in range(self.N)] +
                [r"$\sigma_x^%d\sigma_x^{%d} + \sigma_y^%d\sigma_y^{%d}$"
                 % (n, n, n + 1, n + 1) for n in range(self.N - 1)])

    def adjacent_gates(self, qc):
        return super(LinearSpinChain, self).adjacent_gates(qc, "linear")


class CircularSpinChain(SpinChain):
    """
    A processor based on the physical implementation of
    a circular spin chain qubits system.
    The available Hamiltonian of the system is predefined.
    The processor can simulate the evolution under the given
    control pulses either numerically or analytically.

    Parameters
    ----------
    N: int
        The number of qubits in the system.

    correct_global_phase: float
        Save the global phase, the analytical solution
        will track the global phase.
        It has no effect on the numerical solution.

    sx: int or list
        The delta for each of the qubits in the system.

    sz: int or list
        The epsilon for each of the qubits in the system.

    sxsy: int or list
        The interaction strength for each of the qubit pair in the system.

    t1: list or float, optional
        Characterize the decoherence of amplitude damping for
        each qubit.

    t2: list of float, optional
        Characterize the decoherence of dephasing for
        each qubit.
    """
    def __init__(self, N, correct_global_phase=True,
                 sx=0.25, sz=1.0, sxsy=0.1, t1=None, t2=None):

        super(CircularSpinChain, self).__init__(
            N, correct_global_phase=correct_global_phase,
            sx=sx, sz=sz, sxsy=sxsy, t1=t1, t2=t2)
        self.set_up_params(sx=sx, sz=sz, sxsy=sxsy)
        self.set_up_ops(N)

    def set_up_ops(self, N):
        super(CircularSpinChain, self).set_up_ops(N)
        x = [identity(2)] * N
        x[0] = x[N - 1] = sigmax()
        y = [identity(2)] * N
        y[0] = y[N - 1] = sigmay()
        self.ctrls.append(tensor(x) + tensor(y))

    def set_up_params(self, sx, sz, sxsy):
        # Doc same as in the parent class
        super(CircularSpinChain, self).set_up_params(sx, sz)
        sxsy_para = self._para_list(sxsy, self.N)
        self._paras["sxsy"] = sxsy_para

    @property
    def sxsy_ops(self):
        return self.ctrls[2*self.N: 3*self.N]

    @property
    def sxsy_u(self):
        return self.coeffs[2*self.N: 3*self.N]

    def load_circuit(self, qc):
        return super(CircularSpinChain, self).load_circuit(qc, "circular")

    def get_ops_labels(self):
        return ([r"$\sigma_x^%d$" % n for n in range(self.N)] +
                [r"$\sigma_z^%d$" % n for n in range(self.N)] +
                [r"$\sigma_x^%d\sigma_x^{%d} + \sigma_y^%d\sigma_y^{%d}$"
                 % (n, n, (n + 1) % self.N, (n + 1) % self.N)
                 for n in range(self.N)])

    def adjacent_gates(self, qc):
        return super(CircularSpinChain, self).adjacent_gates(qc, "circular")


class SpinChainGateDecomposer(GateDecomposer):
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

    setup: string
        "linear" or "circular" for two sub-calsses.

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

    setup: string
        "linear" or "circular" for two sub-calsses.

    global_phase: bool
        Record of the global phase change and will be returned.
    """
    def __init__(self, N, params, setup, global_phase, num_ops):
        super(SpinChainGateDecomposer, self).__init__(
            N=N, params=params, num_ops=num_ops)
        self.gate_decomps = {"ISWAP": self.iswap_dec,
                             "SQRTISWAP": self.sqrtiswap_dec,
                             "RZ": self.rz_dec,
                             "RX": self.rx_dec,
                             "GLOBALPHASE": self.globalphase_dec
                             }
        self.N = N
        self._sx_ind = list(range(0, N))
        self._sz_ind = list(range(N, 2*N))
        if setup == "circular":
            self._sxsy_ind = list(range(2*N, 3*N))
        elif setup == "linear":
            self._sxsy_ind = list(range(2*N, 3*N-1))
        self.global_phase = global_phase

    def decompose(self, gates):
        tlist, coeffs = super(SpinChainGateDecomposer, self).decompose(gates)
        return tlist, coeffs, self.global_phase

    def rz_dec(self, gate):
        """
        Decomposer for the RZ gate
        """
        pulse = np.zeros(self.num_ops)
        q_ind = gate.targets[0]
        g = self.params["sz"][q_ind]
        pulse[self._sz_ind[q_ind]] = np.sign(gate.arg_value) * g
        t = abs(gate.arg_value) / (2 * g)
        self.dt_list.append(t)
        self.coeff_list.append(pulse)

    def rx_dec(self, gate):
        """
        Decomposer for the RX gate
        """
        pulse = np.zeros(self.num_ops)
        q_ind = gate.targets[0]
        g = self.params["sx"][q_ind]
        pulse[self._sx_ind[q_ind]] = np.sign(gate.arg_value) * g
        t = abs(gate.arg_value) / (2 * g)
        self.dt_list.append(t)
        self.coeff_list.append(pulse)

    def iswap_dec(self, gate):
        """
        Decomposer for the ISWAP gate
        """
        pulse = np.zeros(self.num_ops)
        q1, q2 = min(gate.targets), max(gate.targets)
        g = self.params["sxsy"][q1]
        if q1 == 0 and q2 == self.N - 1:
            pulse[self._sxsy_ind[self.N - 1]] = -g
        else:
            pulse[self._sxsy_ind[q1]] = -g
        t = np.pi / (4 * g)
        self.dt_list.append(t)
        self.coeff_list.append(pulse)

    def sqrtiswap_dec(self, gate):
        """
        Decomposer for the SQRTISWAP gate
        """
        pulse = np.zeros(self.num_ops)
        q1, q2 = min(gate.targets), max(gate.targets)
        g = self.params["sxsy"][q1]
        if q1 == 0 and q2 == self.N - 1:
            pulse[self._sxsy_ind[self.N - 1]] = -g
        else:
            pulse[self._sxsy_ind[q1]] = -g
        t = np.pi / (8 * g)
        self.dt_list.append(t)
        self.coeff_list.append(pulse)

    def globalphase_dec(self, gate):
        """
        Decomposer for the GLOBALPHASE gate
        """
        self.global_phase += gate.arg_value
