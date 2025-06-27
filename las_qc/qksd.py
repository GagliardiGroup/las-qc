"""
Methods for Quantum Krylov Subspace Diagnolization (QKSD) for LAS
""" 

import logging

import numpy as np
import numpy.linalg as LA
import numpy.typing as npt
from scipy.sparse.linalg import expm_multiply

# TODO: Replace this with some PySCF compatible code
log = logging.getLogger(__name__)

###################   Type Defs    ###################
 
NPComplex = npt.NDArray[np.complex128]


################### Helper Methods ###################

def time_propogate(init_statevector: NPComplex, Hcsc: NPComplex, dt: float, nstates: int) -> tuple[list[NPComplex], list[NPComplex]]:
    """Time evolution of `statevector` with Hcsc in time

    Args:
        init_statevector: trial wavefunction
        Hcsc: Hamultonian
        dt: timestep
        nstats: Number of timesteps

    returns:
        NPComplex: states
        NPComplex: state Omega products
    """

    # Define out Krylov subspace comprised of `time_step` of elements
    # U |\phi_0>
    statevectors = expm_multiply(
        -1j * Hcsc * dt, init_statevector, start=0.0, stop=nstates - 1, num=nstates
    )
    omega_list = [np.asarray(state, dtype=complex) for state in statevectors]

    # Extract Hamiltonians
    # V U |\phi_0>
    H_omega_list = [Hcsc.dot(omega) for omega in omega_list]

    return omega_list, H_omega_list



def qksd_energy(F_mat: NPComplex, S_mat: NPComplex, Stol: float=1e-12, trim: int | None = None) -> np.float64:
    """Calculates the energy based on F and S matrices

    Args:
        F_mat: Complex matrix
        S_mat: Complex matrix

    Kwargs:
        Stol: threshhold for S matrix preconditionign
        trim: Size of krylov subspace to use
              default to using all of `F`

    returns: Hamultonian energy
    """

    # Allow the F_matrix to be trimmed for debuggin
    if trim is None: # Use whole matrix for calclationVh
        m = F_mat.shape[0] # n_timesteps
    else:
        m = trim

    # Using an SVD to condition the F matrix
    # Before doing the eigendecomposition
    U, s, _ = LA.svd(S_mat[: m + 1, : m + 1])

    Dtemp = 1 / np.sqrt(s)
    Dtemp[Dtemp**2 > 1 / Stol] = 0

    Xp = U[0 : len(s), 0 : len(Dtemp)] * Dtemp
    Fp = Xp.T.conjugate() @ F_mat[: m + 1, : m + 1] @ Xp

    # Eigenvalues of the conditioned matrix
    eigvals, _ = LA.eig(Fp)

    return eigvals[0].real


def QKSD(init_state: NPComplex, Hs: NPComplex, time_steps: int = 5, tau: float = 0.1) -> tuple[float, NPComplex, NPComplex]:
    """Performs QKSD on Hamiltonian `Hs`

    Args:
        init_state: Initial trial state
        Hs: Trial hamultonian

    Kwargs:
        time_steps: Number of time steps to take
        tau: Size of timesteps

    Returns:
        QKSD energy
        F array
        S array
    """

    # Create Krylov States
    omega_list, H_omega_list = time_propogate(init_state, Hs, tau, time_steps)

    # Initialize F, S matrices
    # Re-used in each step. but not additive
    F_mat = np.zeros((time_steps, time_steps), dtype=complex)
    S_mat = np.zeros((time_steps, time_steps), dtype=complex)

    # Fill in the S and F matrices
    for m in range(time_steps):
        for n in range(m + 1):
            # < \phi_0 | U_m^+ U_n | \phi_0 >
            Smat_el = np.vdot(omega_list[m], omega_list[n])
            log.debug("S_{}_{} = {}".format(m, n, Smat_el))

            S_mat[m][n] = Smat_el
            S_mat[n][m] = np.conj(Smat_el)

            # Filling the F matrix
            # < \phi_0 | U_m^+ V U_n | \phi_0 >
            Fmat_el = np.vdot(omega_list[m], H_omega_list[n])
            log.debug("F_{}_{} = {}".format(m, n, Fmat_el))

            F_mat[m][n] = Fmat_el
            F_mat[n][m] = np.conj(Fmat_el)

        # qksd_energy costs CPU so check logger first
        if log.level < logging.INFO:
            step_energy = qksd_energy(F_mat, S_mat, trim=m)
            log.debug(f"Energy @ T = {m*tau}: {step_energy}")
            log.debug(f"Energy @ T = {m*tau: .3f}: {step_energy}")

    energy = qksd_energy(F_mat, S_mat)
    log.info("Final QKSD Energy: {energy} Eh")

    return energy, F_mat, S_mat
