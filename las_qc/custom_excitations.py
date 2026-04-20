from typing import Any, List, Tuple

import numpy as np
from mrh.exploratory.citools import grad
from mrh.exploratory.unitary_cc import lasuccsd

# Gets all acceptable operators for UCCSD excluding intra-fragment ones


def custom_excitations(
    num_spatial_orbitals: int,
    num_particles: Tuple[int, int],
    num_sub: List[int],
) -> List[Tuple[Tuple[Any, ...], ...]]:
    """
    Generate a list of fermionic excitation tuples for a LAS-UCCSD ansatz.
    Parameters
    ----------
    num_spatial_orbitals : int
        Total number of spatial orbitals in the active space.
    num_particles : Tuple[int, int]
        Number of alpha and beta electrons.
    num_sub : List[int]
        List specifying the number of orbitals in each LAS fragment.

    Returns
    -------
        A list of excitation tuples
    """
    excitations = []
    norb = num_spatial_orbitals
    uop = lasuccsd.gen_uccsd_op(norb, num_sub)
    a_idxs = uop.a_idxs
    i_idxs = uop.i_idxs
    for a, i in zip(a_idxs, i_idxs):
        excitations.append((tuple(i), tuple(a[::-1])))

    return excitations


def generate_uscc_excitations(
    num_spatial_orbitals: int,
    num_particles: Tuple[int, int],
    num_sub: List[int],
    las=None,
    epsilon=0.0,
    verbose=1,
):
    """
    Generate selected LAS-UCC excitations using gradient screening.
    Parameters
    ----------
    num_spatial_orbitals : int
        Total number of spatial orbitals in the active space.
    num_particles : Tuple[int, int]
        Number of alpha and beta electrons. Included for compatibility
    num_sub : List[int]
        List specifying the number of orbitals in each LAS fragment.
    las : object, optional
        LASSCF object
    epsilon : float, optional
        Threshold for selecting excitations based on gradient magnitude.
        Only excitations with gradients larger than this value are kept.
        Default is 0.0 (LASUCC).
    verbose : int, optional
        Verbosity level controlling printed output. Default is 1. Need to be added later.

    Returns
    -------
    List[Tuple[Tuple[Any, ...], ...]]
        A list of selected excitation tuples
    """
    excitations = []
    all_g, g_sel, a_idxs_new, i_idxs_new = grad.get_grad_exact(las, epsilon=epsilon)
    np.save("all_g.npy", all_g)
    np.save("g_sel.npy", g_sel)

    print("All gradients from exact method = ", all_g)

    for a, i in zip(a_idxs_new, i_idxs_new):
        excitations.append((tuple(i), tuple(a[::-1])))

    return excitations
