from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit import Aer, IBMQ  # import the Aer and IBMQ providers
from qiskit.providers.aer import noise  # import Aer noise models

# Choose a real device to simulate
IBMQ.load_accounts() # this wont work, fix later by adding the API token on account
device = IBMQ.get_backend('ibmq_16_melbourne')
properties = device.properties()
coupling_map = device.configuration().coupling_map

# Generate an Aer noise model for device
noise_model = noise.device.basic_device_noise_model(properties)
basis_gates = noise_model.basis_gates

# Generate a quantum circuit
q = QuantumRegister(2)
c = ClassicalRegister(2)
qc = QuantumCircuit(q, c)

qc.h(q[0])
qc.cx(q[0], q[1])
qc.measure(q, c)

# Perform noisy simulation
backend = Aer.get_backend('qasm_simulator')
job_sim = execute(qc, backend,
                  coupling_map=coupling_map,
                  noise_model=noise_model,
                  basis_gates=basis_gates)
sim_result = job_sim.result()

print(sim_result.get_counts(qc))