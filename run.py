from QuantumCircuit import *

# Qubit frequency in (GHz)
omega_1 = 2*np.pi * 5.708390
omega_2 = 2*np.pi * 5.202204

# Anharmonicity in (GHz)
alpha_1 = - 2*np.pi * 0.261081457
alpha_2 = - 2*np.pi * 0.275172227

# Longitudinal decoherence in (ns)
T1_1 = 25.111199 * 1e3
T1_2 = 17.987260 * 1e3

# Transveral decoherence in (ns)
T2_1 = 23.872090 * 1e3
T2_2 = 10.778744 * 1e3

# Construct Qubit Objects
q1 = Qubit(omega_1, alpha_1, T1_1, T2_1)
q2 = Qubit(omega_2, alpha_2, T1_2, T2_2)

# Coupling in (GHz)
J = 2*np.pi * 3.8*1e-3

# cpl_map[[i, j], [k, l]]: i-th (k-th) and j-th (l-th) qubits are coupled
cpl_map = [[0, 1]]

# cpl_list[i]: coupling strength of i-th couple (defined by cpl_map)
cpl_list = [J]

# Create Quantum Circuit
qc = QuantumCircuit([q1, q2],cpl_map=cpl_map,cpl=cpl_list)

# Do a Rx gate on the first qubit
qc.rx(0,np.pi)

# Do a CZ gate between the first and second qubit
qc.cz(0,1)

qc.state_vector()

# Simulate the circuit
result = run(qc)
