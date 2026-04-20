# Qiskit imports
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper

def get_hamiltonian(frag, nelecas_sub, ncas_sub, h1, h2, mapper=JordanWignerMapper()):
    """Give an option to what if frag is None"""
    if frag is not None:
        h1 = h1[frag]
        h2 = h2[frag]

    electronic_energy = ElectronicEnergy.from_raw_integrals(h1, h2)

    # Choose fermion-to-qubit mapping
    hamiltonian = mapper.map(electronic_energy.second_q_op())
    return hamiltonian
