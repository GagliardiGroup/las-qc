# Qiskit imports
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.properties import ParticleNumber
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper


def get_hamiltonian(frag, nelecas_sub, ncas_sub, h1, h2):
    if frag is None:
        num_alpha = nelecas_sub[0]
        num_beta = nelecas_sub[1]
        n_so = ncas_sub*2
    else:
        # Get alpha and beta electrons from LAS
        num_alpha = nelecas_sub[frag][0]
        num_beta = nelecas_sub[frag][1]
        n_so = ncas_sub[frag]*2
        h1 = h1[frag]
        h2 = h2[frag]

    # For QPE, need second_q_ops
    # Hacking together an ElectronicStructureDriverResult to create second_q_ops
    # Lines below stolen from qiskit's FCIDump driver and modified
    particle_number = ParticleNumber(
        num_spin_orbitals=n_so,
        num_particles=(num_alpha, num_beta),
    )

    # Assuming an RHF reference for now, so h1_b, h2_ab, h2_bb are created using 
    # the corresponding spots from h1_frag and just the aa term from h2_frag
    electronic_energy = ElectronicEnergy.from_raw_integrals(h1, h2)

    # QK NOTE: under Python 3.6, pylint appears to be unable to properly identify this case of
    # nested abstract classes (cf. https://github.com/Qiskit/qiskit-nature/runs/3245395353).
    # However, since the tests pass I am adding an exception for this specific case.
    # pylint: disable=abstract-class-instantiated
    driver_result = ElectronicStructureProblem()
    driver_result.add_property(electronic_energy)
    driver_result.add_property(particle_number)

    second_q_ops = driver_result.second_q_ops()

    # Choose fermion-to-qubit mapping
    qubit_converter = QubitConverter(mapper = JordanWignerMapper(), two_qubit_reduction=False)
    hamiltonian = JordanWignerMapper().map(electronic_energy.second_q_op())#qubit_ops[0]
    return hamiltonian

