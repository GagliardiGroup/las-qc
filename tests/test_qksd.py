import unittest

import numpy as np

from las_qc.qksd import time_propogate, qksd_energy, QKSD, log

from pyscf import gto, lib, scf, mcscf, ao2mo

import numpy as np
from scipy.sparse import csc_matrix

# PySCF imports
from pyscf import gto, lib, scf, mcscf, ao2mo

# Qiskit imports
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library.initial_states import HartreeFock
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_aer import AerSimulator
from qiskit import transpile


class QKSDUnitTests(unittest.TestCase):
    """Unit test for QKSD and support methods"""

    def test_time_propogate(self):
        """Tests time propogation code"""
        state = np.ones(3) / np.sqrt(3)
        H = np.array(
            [
                [1.0, 0.2, 0.1],
                [0.2, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ]
        )

        # Calculate states
        states, H_omegas = time_propogate(state, H, 0.2, 4)

        # Generated from running code. Assuming Ruhee's version is correct
        # Imaginary time evolution is imaginary
        STATE_FIXTURE = [
            np.array([0.57735027 + 0.0j, 0.57735027 + 0.0j, 0.57735027 + 0.0j]),
            np.array(
                [
                    0.55772056 - 0.14837364j,
                    0.55531479 - 0.15966076j,
                    0.55772056 - 0.14837364j,
                ]
            ),
            np.array(
                [
                    0.50020633 - 0.28646985j,
                    0.49081021 - 0.30751038j,
                    0.50020633 - 0.28646985j,
                ]
            ),
            np.array(
                [
                    0.40883503 - 0.40473561j,
                    0.38852733 - 0.43258816j,
                    0.40883503 - 0.40473561j,
                ]
            ),
        ]

        for test, ref in zip(states, STATE_FIXTURE):
            np.testing.assert_almost_equal(test, ref)

        # Generated from running code. Assuming Ruhee's version is correct
        H_OMEGA_FIXTURE = [
            np.array([0.75055535 + 0.0j, 0.80829038 + 0.0j, 0.75055535 + 0.0j]),
            np.array(
                [
                    0.72455557 - 0.19514316j,
                    0.77840302 - 0.21901022j,
                    0.72455557 - 0.19514316j,
                ]
            ),
            np.array(
                [
                    0.64838901 - 0.37661891j,
                    0.69089274 - 0.42209832j,
                    0.64838901 - 0.37661891j,
                ]
            ),
            np.array(
                [
                    0.527424 - 0.53172681j,
                    0.55206134 - 0.5944824j,
                    0.527424 - 0.53172681j,
                ]
            ),
        ]

        for test, ref in zip(H_omegas, H_OMEGA_FIXTURE):
            np.testing.assert_almost_equal(test, ref)

    def test_qksd_energy(self):
        # Random matrices for testing
        F = np.array(
            [
                [2.0, 1.1, 3.3],
                [2.1, 4.0, 1.2],
                [3.3, 1.2, 1.4],
            ]
        )

        S = np.array(
            [
                [1.0, 0.1, 0.1],
                [0.1, 1.0, 0.1],
                [0.1, 0.1, 1.0],
            ]
        )

        energy = qksd_energy(F, S)
        self.assertAlmostEqual(5.425059120768067, energy)

        # Test trimming (for debugging)
        energy = qksd_energy(F, S, trim=1)
        self.assertAlmostEqual(4.4440882788365865, energy)


class QDKSIntegratinoTests(unittest.TestCase):
    """Integration tests for QKSD

    TODO: Make a suite of standard structures to test with
    TODO: Make a standard suite of LAS calculations (module init)
    """

    def test_HF(self):
        """Test QKSD with HF-derived state

        This will have to be redon once we have a test-structure suite
        """
        JWmapper = JordanWignerMapper()

        xyz = """H 0.0 0.0 0.0
                 H  0.0 0.0 0.5
                 H  0.0 0.0 1.5
                 H  0.0 0.0 2.0"""

        # Perform an RHF calculation using PySCF
        mol = gto.M(
            atom=xyz,
            basis="sto-3g",
            charge=0,
            spin=0,
            symmetry=False,
            output="/dev/null",
            verbose=lib.logger.DEBUG,
        )
        mf = scf.RHF(mol).newton()
        hf_en = mf.kernel()
        nuc_rep_en = mf.energy_nuc()

        # Set up CASCI for active orbitals
        mc = mcscf.CASCI(mf, 4, (2, 2))
        n_so = mc.ncas
        (n_alpha, n_beta) = (mc.nelecas[0], mc.nelecas[1])

        # Extract and convert the 1 and 2e integrals
        # To obtain the qubit Hamiltonian
        h1, e_core = mc.h1e_for_cas()
        h2 = ao2mo.restore(1, mc.get_h2eff(), mc.ncas)



        electronic_energy = ElectronicEnergy.from_raw_integrals(h1, h2)
        second_q_op = electronic_energy.second_q_op()

        hamiltonian = JWmapper.map(second_q_op)
        # print(hamiltonian)

        # Create a unitary by exponentiating the Hamiltonian
        # Using the scipy sparse matrix form
        ham_mat = hamiltonian.to_matrix()
        Hsp = csc_matrix(ham_mat, dtype=complex)


        # Create a Hartree-Fock state circuit
        init_state = HartreeFock(n_so, (n_alpha, n_beta), JWmapper)
        init_state.save_statevector()

        # Run in simulator
        simulator = AerSimulator(method="statevector")
        init_state = transpile(init_state, simulator)
        job_result = simulator.run(init_state, shots=1, memory=True).result()
        init_statevector = np.asarray(
            job_result.get_statevector(init_state)._data, dtype=complex
        )

        # Do QKSD
        import logging
        import sys
        log.setLevel(0)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        log.addHandler(handler)
        log.info("HI. Why do you do this to me?")
        energy, _, __ = QKSD(init_statevector, Hsp)

       # Cleanup after PySCF
        mol.stdout.close()
        
