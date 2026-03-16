#########################
# LASUCC Solver Class
# PySCF-way standalone wrapper for LAS-VQE
#########################

import numpy as np
from mrh.exploratory.citools import grad
from mrh.exploratory.unitary_cc import lasuccsd
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit_nature.second_q.mappers import JordanWignerMapper

from las_qc.custom_UCC import custom_UCC

from .lasqc import LASQC


class LASUCC(LASQC):

    def __init__(self, mol, *, epsilon=0.0, **kwargs):
        super().__init__(mol, **kwargs)

        self.init_state = None
        self.mapped_ham = None
        self.ansatz = None
        self.e_tot = None
        self.epsilon = epsilon

    def _custom_excitations(self, num_spin_orbitals, num_particles, num_sub, eps=0.0):
        '''Give an option for full list or selected list of excitations for USCC; must be moved to custom_UCC file---  !!! this needs to be defined outside due to a circular import'''
        if eps==0.0:
            excitations = []
            norb = int(num_spin_orbitals / 2)
            uop = lasuccsd.gen_uccsd_op(norb, num_sub)
            a_idxs = uop.a_idxs
            i_idxs = uop.i_idxs
            for a, i in zip(a_idxs, i_idxs):
                excitations.append((tuple(i), tuple(a[::-1])))

        else:
            a_sel, i_sel = grad.get_grad_select(self.las, eps=0.0) # Add the USCC part here

        return excitations


    def generate_ansatz(self, init_state):
        ansatz = custom_UCC(
            num_spatial_orbitals=self.las.ncas,
            num_particles=self.las.nelecas,
            excitations="selected",
            qubit_mapper=JordanWignerMapper(),
            initial_state=init_state,
            epsilon=self.epsilon,
            preserve_spin=False,
            las=self.las,
        )
        return ansatz

    def run(self, statevectors=None, anstaz=None, estimator=None, optimizer=None):
        super().run()

        print("[LASUCC] Running LAS-UCC with VQE...")
        
        self.ansatz = self.generate_ansatz(self.init_state) # add verbose

        optimizer = L_BFGS_B(maxfun=10000, iprint=101)
        init_pt = np.zeros(self.ansatz.num_parameters)

        estimator = AerEstimator() # need to update EstimatorV2!

        algorithm = VQE(
            ansatz=self.ansatz,
            optimizer=optimizer,
            estimator=estimator,
            initial_point=init_pt
        )
        #print ("UCC = ", self.mapped_ham)
        result = algorithm.compute_minimum_eigenvalue(self.mapped_ham)

        self.e_tot = result.eigenvalue.real + self.las.h1e_for_cas()[1]

        print("[LASUCC] Final LAS-UCC energy:", self.e_tot)
        print ("VQE result:")
        print (result)
        return self.e_tot

