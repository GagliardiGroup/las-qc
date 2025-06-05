#########################
# LASUCC Solver Class
# PySCF-way standalone wrapper for LAS-VQE
#########################

import numpy as np
from qiskit import Aer, QuantumCircuit, QuantumRegister
from qiskit.utils import QuantumInstance
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import L_BFGS_B

from custom_UCC import custom_UCC
from get_hamiltonian import get_hamiltonian
from mrh.exploratory.unitary_cc import lasuccsd

class LASUCC:
    def __init__(self, las):
        self.las = las
        self.mol = las.mol
        self.mo_coeff = las.mo_coeff
        self.e_las = las.e_tot
        self.nmo = las.mo_coeff.shape[1]
        self.ncas = las.ncas
        self.ncore = las.ncore
        self.nocc = self.ncas + self.ncore
        self.nelecas = las.nelecas
        self.ncas_sub = las.ncas_sub

    def _custom_excitations(self, num_spin_orbitals, num_particles, num_sub):
        excitations = []
        norb = int(num_spin_orbitals / 2)
        uop = lasuccsd.gen_uccsd_op(norb, num_sub)
        a_idxs = uop.a_idxs
        i_idxs = uop.i_idxs
        for a, i in zip(a_idxs, i_idxs):
            excitations.append((tuple(i), tuple(a[::-1])))
        return excitations

    def initialize_fragments(self,mapper=JordanwignerMapper(), estimator, optimizer):
        '''Initializes LAS fragments'''
        self.mapper = mapper

        init_state = #SV need to define a method for intializing LAS

        h1, e_core = self.las.h1e_for_cas()
        h2 = lib.numpy_helper.unpack_tril (las.get_h2eff().reshape (nmo*ncas,ncas*(ncas+1)//2)).reshape(nmo, ncas, ncas, ncas)[ncore:nocc,:,:,:])
        hamiltonian = get_hamiltonian (None, self.nelecas, self.ncas, h1, h2)

        return init_state, hamiltonian

    def generate_anstaz(self, init_state):
        return anstaz

    def kernel(self, statevectors=None, anstaz=None, estimator=None):
        print("[LASUCC] Running LAS-UCC with VQE...")

        h1, e_core = self.las.h1e_for_cas()
        h2 = lib.numpy_helper.unpack_tril (las.get_h2eff().reshape (nmo*ncas,ncas*(ncas+1)//2)).reshape (nmo, ncas, ncas, ncas)[ncore:nocc,:,:,:]
        hamiltonian = get_hamiltonian (None, self.nelecas, self.ncas, h1, h2)

        n_qubits = np.sum(self.ncas_sub) * 2
        
        self.init_state, self.ham = self.initialize_fragments(mapper,estimator, optimizer)
        self.ansatz = self.generate_anstaz(self.init_state) # add verbose
        

        ansatz = custom_UCC(
            num_particles=(2, 2),
            num_spin_orbitals=n_qubits,
            excitations=self._custom_excitations,
            initial_state=init_circ,
            preserve_spin=False
        )

        optimizer = L_BFGS_B(maxfun=10000, iprint=101)
        init_pt = np.zeros(ansatz.num_parameters)
        instance = QuantumInstance(backend=Aer.get_backend('aer_simulator'), shots=shots)

        algorithm = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            quantum_instance=instance,
            initial_point=init_pt
        )

        result = algorithm.compute_minimum_eigenvalue(hamiltonian)
        self.e_tot = result.eigenvalue.real
        print("[LASUCC] Final LAS-UCC energy:", self.e_tot)
        return self.e_tot
