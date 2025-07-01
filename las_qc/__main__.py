from pyscf import gto, lib

from .geometry import get_geom
from .lasqc import LASQC


def main():
    """Helper function for running LAS-QC as a script"""
    xyz = get_geom("far")
    mol = gto.M(
        atom=xyz,
        basis="sto-3g",
        output="h4_sto3g.log",
        symmetry=False,
        verbose=lib.logger.DEBUG,
    )
    _lasqc_wfn = LASQC(
        mol,
        frag_orbs=(2, 2),
        frag_elec=(2, 2),
        frag_atom_list=((0, 1), (2, 3)),
        spin_sub=(1, 1),
    )


if __name__ == "__main__":
    # TODO: Parse commandline arguments
    main()
