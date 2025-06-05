#######################
# Define LASQC class
# Set up and run a LASSCF calculation
# Other methods as solvers for LASQC
#########################

# PySCF imports
from pyscf import gto, scf, lib
# mrh imports
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.mcscf.lasci import h1e_for_las

from get_geom import get_geom

# Define LASQC class
class LASQC:
    def __init__(self, mol, las=None, mf=None, frag_orbs=None, frag_elec=None, frag_atom_list=None, spin_sub=None):

        if mf is None:
            # Do RHF
            mf = scf.RHF(mol).run()
            print("HF energy: ", mf.e_tot)
            
        if las is None:
            # Create LASSCF object
            # Keywords: (wavefunction obj, num_orb in each subspace, (nelec in each subspace)/((num_alpha, num_beta) in each subspace), spin multiplicity in each subspace)
            las = LASSCF(mf, frag_orbs, frag_elec, spin_sub=spin_sub)

            # Localize the chosen fragment active spaces
            loc_mo_coeff = las.localize_init_guess(frag_atom_list, mf.mo_coeff)

            # Run LASSCF
            las.kernel(loc_mo_coeff)
            print("LASSCF energy: ", las.e_tot)

        self.mf = mf
        self.las = las

if __name__ == '__main__':

    xyz = get_geom('far')
    mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log',
        symmetry=False, verbose=lib.logger.DEBUG)
    lasqc_wfn = LASQC(mol, frag_orbs=(2,2), frag_elec=(2,2), frag_atom_list=((0,1),(2,3)), spin_sub=(1,1))

    # Choose one post-LAS solver
    solver = LASUSCC(lasqc.las)  # or LASUCC, LASQKSD
    solver.kernel()

