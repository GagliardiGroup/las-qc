#######################
# Define LASQC class
# Set up and run a LASSCF calculation
# Other methods as solvers for LASQC
#########################

from qiskit import Aer, QuantumCircuit, QuantumRegister
from qiskit.utils import QuantumInstance
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper

# PySCF imports
from pyscf import scf

# mrh imports
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

import initialize_fragments as initf


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

        self.mf = mf
        self.las = las
        self.mol = las.mol
        self.mo_coeff = las.mo_coeff
        self.nmo = las.mo_coeff.shape[1]
        self.ncas = las.ncas
        self.ncore = las.ncore
        self.nocc = self.ncas + self.ncore
        self.nelecas = las.nelecas
        self.ncas_sub = las.ncas_sub


    def initialize_fragments(self, method='DI', **kwargs):
        '''Initializes LAS fragments: this should call something from initialize_fragments.py file'''
        
        match method.lower():
            case 'di': init_fn = initf.direct_initialization
            case 'sf': init_fn = initf.spectral_filtering
            case 'qpe': init_fn = initf.qpe_initialization
            case 'vqe': init_fn = initf.vqe_initialization

        self.init_state = init_fn(**kwargs) 
        

    def get_mapped_hamiltonian(self):
        h1, e_core = self.las.h1e_for_cas()
        h2 = lib.numpy_helper.unpack_tril (self.las.get_h2eff().reshape (self.nmo*self.ncas,self.ncas*(self.ncas+1)//2)).reshape(self.nmo, self.ncas, self.ncas, self.ncas)[self.ncore:self.nocc,:,:,:])
        hamiltonian = get_hamiltonian (None, self.nelecas, self.ncas, h1, h2)
        return hamiltonian

    def run(self):
        '''common things for all methods '''
        if self.init_state is None:
            self.init_state = initialize_fragments(...)

        if self.mapped_ham is None:
            self.mapped_ham = get_mapped_hamiltonian()

        result_dict  = energy, circuit, gates

        if self.__class__ is LASQC:
            raise NotImplementedError ("run method not implemented")
        return result_dict

if __name__ == '__main__':

    xyz = get_geom('far')
    mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log',
        symmetry=False, verbose=lib.logger.DEBUG)
    lasqc_wfn = LASQC(mol, frag_orbs=(2,2), frag_elec=(2,2), frag_atom_list=((0,1),(2,3)), spin_sub=(1,1))

    # Choose one post-LAS solver
    solver = LASUSCC(lasqc.las)  # or LASUCC, LASQKSD
    solver.kernel()

