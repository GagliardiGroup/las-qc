from typing import Tuple, List, Any
import numpy as np

from pyscf import mcscf, lib
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.exploratory.citools import grad

# Gets all acceptable operators for UCCSD excluding intra-fragment ones


def custom_excitations(
    num_spatial_orbitals: int,
    num_particles: Tuple[int, int],
    num_sub: List[int],
) -> List[Tuple[Tuple[Any, ...], ...]]:
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
    verbose=1
):
    excitations = []
    all_g, g_sel, a_idxs_new, i_idxs_new = grad.get_grad_exact(las, epsilon=epsilon) # new lasuscc code in mrh
    np.save('all_g.npy', all_g)
    np.save('g_sel.npy', g_sel)

    print("All gradients from exact method = ", all_g)

    for a, i in zip(a_idxs_new, i_idxs_new):
        excitations.append((tuple(i), tuple(a[::-1])))

    return excitations
