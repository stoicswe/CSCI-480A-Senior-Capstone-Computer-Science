# single_q_measurement.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, BasicAer # using BasicAer package instead of Aer due to import error

# Define the Quantum and Classical Registers
q = QuantumRegister(1)
c = ClassicalRegister(1)

# Build the circuit
single_q_measurement = QuantumCircuit(q, c)
single_q_measurement.measure(q, c)
 
# Execute the circuit
job = execute(single_q_measurement, backend = BasicAer.get_backend('qasm_simulator'), shots=1024)
result = job.result()

# Print the result
print(result.get_counts(single_q_measurement))