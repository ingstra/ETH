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
        #print('num_ops = ', num_ops)
        self.global_phase = global_phase

    def decompose(self, layers):
        # TODO further improvement can be made here,
        # e.g. merge single qubit rotation gate, combine XX gates etc.
        self.dt_list = [[] for i in range(len(layers))]
        self.coeff_list = [[] for i in range(self.num_ops)]
        temp = []
        for layer_idx,layer in enumerate(layers):
            #print('layer_idx',layer_idx)
            for gate in layer:
                if gate.name not in self.gate_decomps:
                    raise ValueError("Unsupported gate %s" % gate.name)
                self.gate_decomps[gate.name](gate,layer_idx)
            #print('self.coeff_list',self.coeff_list)
            coeff_len = 0
            for i in range(self.num_ops):
                try:
                    if len(self.coeff_list[i][layer_idx]) > coeff_len:
                        coeff_len = len(self.coeff_list[i][layer_idx])
                except:
                    0

            for i in range(self.num_ops):
                try:
                    if len(self.coeff_list[i][layer_idx]) < coeff_len:
                        self.coeff_list[i][layer_idx].extend([0] * (coeff_len-len(self.coeff_list[i][layer_idx])))
                except:
                    self.coeff_list[i].append([0] * coeff_len)

            #print('self_coeff',self.coeff_list)
            # Get the longest list time list
            #temp = max((x) for x in self.dt_list[idx]]
            temp.extend(max(self.dt_list[layer_idx], key = len)) # this is the time for layer 1
            #print('coeff_list',self.coeff_list)
            #print('temp',temp)

        #print('dt_list',self.dt_list[0][0])


        # Have some layer stuff
        #print('shape self.coeff_list',np.shape(self.coeff_list))
        #coeffs = np.vstack(self.coeff_list)
        tlist = np.empty(len(temp))


        #tlist = self. dt_list
        #coeffs = self.coeff_list
        #print('tlist',tlist)
        #print('coeffs',self.coeff_list)
        #print(np.shape(tlist))
        #print(np.shape(self.coeff_list))

        t = 0
        #tlist = np.empty(len(self.dt_list)+1)
        for i in range(len(temp)):
            tlist[i] = t
            t += temp[i]

        #print('coeffs_shape',coeffs)
        #print('coeffs_shape',np.shape(coeffs))
        #print('tlist',tlist)
        #print('tlist_shape',np.shape(tlist))
        #tlist, coeffs = super(ETHCompiler, self).decompose(gates)

        # Join sublists
        for i in range(self.num_ops):
            self.coeff_list[i] = np.array(sum(self.coeff_list[i], []))

        coeffs = self.coeff_list
        #print('coeffs',np.shape(coeffs))
        #print(np.hstack([[0], tlist]))
        #print(len(tlist))
        return tlist, coeffs, self.global_phase

    def rz_dec(self, gate):
        """
        Compiler for the RZ gate
        """
        # Virtual rz gate
        pulse = np.zeros(self.num_ops)
        q_ind = gate.targets[0]
        # todo

    def rx_dec(self, gate, idx):
        """
        Compiler for the RX gate
        """
        #print('idx',idx)
        # Gaussian width in ns
        sigma = 10
        # Gate time in ns
        L = 50
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

        # Total time in ns
        t_total = L
        #tlist = np.linspace(0,t_total,500)
        tlist = np.linspace(0,t_total,100)

        #dt = tlist[1] - tlist[0]
        #tlist = np.append(tlist,(t_total+dt))
        amp = 0.06345380720748712 * gate.arg_value / np.pi
        args = {'amp': amp, 'qscale': 0.032, 'freq': 0}

        # set pulse to zero
        pulse = np.zeros(len(tlist)-1) # This we should not do!
        q = gate.targets[0] # target qubit
        #print('q',q)

        # in-phase component
        I = H_drive_coeff_x(tlist, args)
        # out of-phase component
        Q = H_drive_coeff_y(tlist, args)

        I = np.delete(I, len(I)-1)
        Q = np.delete(Q, len(Q)-1)
        R = np.vstack((I,Q))
        #print('R',R)

        dt_list = tlist[1:] - tlist[:-1]
        self.dt_list[idx].append(dt_list)

        #pulse[2*q:2*(1+q),:] = R
        #print('pulse',pulse)

        #print(pulse)
        #print(np.shape(pulse))
        self.coeff_list[2*q].append(list(I))
        self.coeff_list[2*q+1].append(list(Q))

        #print(len(self.coeff_list))

        #self.coeff_list = np.concatenate((self.coeff_list,[I,Q]),axis=1)
        #self.coeff_list.append(I)
        #self.coeff_list.append(Q)

    def globalphase_dec(self, gate):
        """
        Compiler for the GLOBALPHASE gate
        """
        self.global_phase += gate.arg_value
