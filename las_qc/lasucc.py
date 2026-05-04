#########################
# LASUCC Solver Class
# PySCF-way standalone wrapper for LAS-VQE
#########################

from typing import Callable
import logging

import numpy as np
from mrh.exploratory.citools import grad
from mrh.exploratory.unitary_cc import lasuccsd
from qiskit.providers import BackendV2
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit_nature.second_q.mappers import JordanWignerMapper

from las_qc.custom_UCC import custom_UCC
from pathlib import Path

from .lasqc import LASQC

# Add a logger
log = logging.getLogger(__name__)


class LASUCC(LASQC):
    def __init__(self, mol, *, epsilon=0.0, **kwargs):
        super().__init__(mol, **kwargs)

        self.init_state = None
        self.mapped_ham = None
        self.ansatz = None
        self.e_tot = None
        self.epsilon = epsilon

    def _custom_excitations(self, num_spin_orbitals, num_particles, num_sub, eps=0.0):
        """Give an option for full list or selected list of excitations for USCC; must be moved to custom_UCC file---  !!! this needs to be defined outside due to a circular import"""
        if eps == 0.0:
            excitations = []
            norb = int(num_spin_orbitals / 2)
            uop = lasuccsd.gen_uccsd_op(norb, num_sub)
            a_idxs = uop.a_idxs
            i_idxs = uop.i_idxs
            for a, i in zip(a_idxs, i_idxs):
                excitations.append((tuple(i), tuple(a[::-1])))

        else:
            a_sel, i_sel = grad.get_grad_select(
                self.las, eps=0.0
            )  # Add the USCC part here

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

    def run(
        self,
        estimator: EstimatorV2 | None = None,
        statevectors=None,
        ansatz=None,
        optimizer=None,
        backend: BackendV2 | None = None,
        pass_manager: PassManager | None = None,
        checkpoint_file: str | None = None,
        callback: Callable | None = None,
    ):
        super().run()

        log.info("[LASUCC] Running LAS-UCC with VQE...")



        if ansatz is None:
            ansatz = self.generate_ansatz(self.init_state)  # add verbose

        # Setup the estimator
        if estimator is None:
            # If no backend is proved, run with QiskitAer
            if backend is None:
                log.warn("No backend provided. Simulating with QiskitAer")
                backend = AerSimulator()

            # Should the user provide this as well?
            estimator = EstimatorV2.from_backend(backend=backend)
        else:
            # Some oddness with the AerEstimatorV2
            if hasattr(estimator, "backend"):
                backend = estimator.backend()
            elif hasattr(estimator, "_backend"):
                backend = estimator._backend
            elif backend is None:
                log.error("Unable to retrieve the backend. Could not transpile")

        # Generate a pass manager to compile our circuits
        if pass_manager is None:
            log.warn("No pass manager provided. Creating default pass manager")
            pass_manager = generate_preset_pass_manager(backend=backend)
        ansatz_isa = pass_manager.run(ansatz)

        # Initialize the optimizer
        if optimizer is None:
            optimizer = L_BFGS_B(maxfun=10000, iprint=101)

        if checkpoint_file:
            print(f"Parameters will be saved to `{checkpoint_file}`")


        if checkpoint_file:
            try:
                init_pt = np.load(checkpoint_file)["params"]
            except FileNotFoundError:
                init_pt = np.zeros(ansatz.num_parameters)
        else:
            init_pt = np.zeros(ansatz.num_parameters)

        init_pt = np.zeros(ansatz.num_parameters)

        # Configure the callback
        def callback(step: int, params, est_val, meta: dict):
            print(f"Step {step}: {est_val}")
            if checkpoint_file is None:
                print(f"  Parameters: {params}")
            else:
                np.savez(
                    checkpoint_file,
                    step=step,
                    est_val=est_val,
                    params=params,
                    meta=meta
                )

        # Run VQE
        algorithm = VQE(
            ansatz=ansatz_isa,
            optimizer=optimizer,
            estimator=estimator,
            initial_point=init_pt,
            callback=callback
        )

        log.info("Running VQE...")
        result = algorithm.compute_minimum_eigenvalue(self.mapped_ham)

        self.e_tot = result.eigenvalue.real + self.las.h1e_for_cas()[1]
        log.info("[LASUCC] Final LAS-UCC energy:", self.e_tot)

        return self.e_tot, result
