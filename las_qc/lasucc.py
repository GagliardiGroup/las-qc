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
from mrh.exploratory.unitary_cc import lasuccsd

from lasqc import LASQC

class LASUCC(LASQC):
    

    def _custom_excitations(self, num_spin_orbitals, num_particles, num_sub, eps=0.0):
        '''Give an option for full list or selected list of excitations for USCC; must be moved to custom_UCC file'''
        if eps==0.0:
            excitations = []
            norb = int(num_spin_orbitals / 2)
            uop = lasuccsd.gen_uccsd_op(norb, num_sub)
            a_idxs = uop.a_idxs
            i_idxs = uop.i_idxs
            for a, i in zip(a_idxs, i_idxs):
                excitations.append((tuple(i), tuple(a[::-1])))

        else:
            get_grad_select() # Add the USCC part here


        return excitations


    def generate_ansatz(self, init_state):
        ansatz = custom_UCC(
            num_particles=(2, 2),
            num_spin_orbitals=n_qubits,
            initial_state=init_circ,
            epsilon=0.0
            preserve_spin=False
        )
        return ansatz

    def run(self, statevectors=None, anstaz=None, estimator=None, optimizer=None):
        super().run()
        print("[LASUCC] Running LAS-UCC with VQE...")


        n_qubits = np.sum(self.ncas_sub) * 2
        
        if self.ansatz is None:
            self.ansatz = self.generate_anstaz(self.init_state) # add verbose


        optimizer = L_BFGS_B(maxfun=10000, iprint=101)
        init_pt = np.zeros(ansatz.num_parameters)

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
