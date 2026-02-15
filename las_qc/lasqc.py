#######################
# Define LASQC class
# Set up and run a LASSCF calculation
# Other methods as solvers for LASQC
#########################

from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit.primitives import Estimator, BaseEstimator
from qiskit_aer.primitives import Estimator as AerEstimator

# PySCF imports
from pyscf import scf, lib, ao2mo, mcscf

# mrh imports
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

import las_qc.initialize_fragments as initf
from las_qc.get_hamiltonian import get_hamiltonian

# Define LASQC class
class LASQC:
    def __init__(self, mol, las=None, mf=None, frag_orbs=None, frag_elec=None, frag_atom_list=None, spin_sub=None):
        
        self.init_state = None
        self.mapped_ham = None

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

        print (mol, las, mf, frag_orbs, frag_elec, frag_atom_list, spin_sub)

        self.mf = mf
        self.las = las
        self.mol = las.mol
        self.mo_coeff = las.mo_coeff
        self.nmo = las.mo_coeff.shape[1]
        self.ncas = las.ncas
        self.ncore = las.ncore
        self.ci = las.ci
        self.nelecas_sub = las.nelecas_sub
        self.nocc = self.ncas + self.ncore
        self.nelecas = las.nelecas
        self.ncas_sub = las.ncas_sub
        self.mc = mcscf.CASCI(mf, las.ncas, las.nelecas)

    def initialize_fragments(self, method='DI', **kwargs):
        """Initializes LAS fragments: this should call something from initialize_fragments.py file"""

        method_map = {
            'di': initf.direct_initialization,
            'sf': initf.spectral_filtering,
            'qpe': initf.qpe_initialization,
            'vqe': initf.vqe_initialization,
            }

        try:
            init_fn = method_map[method.lower()]
        except KeyError:
            raise ValueError(f"Unknown initialization method: {method}")

        self.init_state = init_fn(self, **kwargs)

    def get_mapped_hamiltonian(self):
        h1, e_core = self.las.h1e_for_cas()
        print (h1)
        h2 = ao2mo.restore(
            1, self.mc.get_h2eff(self.mo_coeff), self.mc.ncas
        )
        print (h2.shape)
        hamiltonian = get_hamiltonian (None, self.nelecas, self.ncas, h1, h2)
        return hamiltonian

    def run(self):
        '''common things for all methods '''
        if self.init_state is None:
            self.initialize_fragments(method='di')

        if self.mapped_ham is None:
            self.mapped_ham = self.get_mapped_hamiltonian()

        if self.__class__ is LASQC:
            raise NotImplementedError ("run method not implemented")

if __name__ == '__main__':

    xyz = get_geom('far')
    mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log',
        symmetry=False, verbose=lib.logger.DEBUG)
    lasqc_wfn = LASQC(mol, frag_orbs=(2,2), frag_elec=(2,2), frag_atom_list=((0,1),(2,3)), spin_sub=(1,1))

    # Choose one post-LAS solver
    solver = LASUCC(lasqc.las)  # or LASUCC, LASQKSD
    solver.kernel()

