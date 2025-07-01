# Qiskit imports
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper


def get_hamiltonian(frag, nelecas_sub, ncas_sub, h1, h2, mapper=None):
    if mapper is None:
        JordanWignerMapper()

    if frag is None:
        ...
        # Unused variables are comment
        # num_alpha = nelecas_sub[0]
        # num_beta = nelecas_sub[1]
        # n_so = ncas_sub*2
    else:
        # Unused variables are comment
        # Get alpha and beta electrons from LAS
        # num_alpha = nelecas_sub[frag][0]
        # num_beta = nelecas_sub[frag][1]
        # n_so = ncas_sub[frag]*2
        h1 = h1[frag]
        h2 = h2[frag]

    # Assuming an RHF reference for now, so h1_b, h2_ab, h2_bb are created using
    # the corresponding spots from h1_frag and just the aa term from h2_frag
    electronic_energy = ElectronicEnergy.from_raw_integrals(h1, h2)

    # Choose fermion-to-qubit mapping
    hamiltonian = mapper.map(electronic_energy.second_q_op())  # qubit_ops[0]
    return hamiltonian
