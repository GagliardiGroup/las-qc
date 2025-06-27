'''
Minimal test to make sure nothing breaks
while updating for qiskit version
Checking against hardcoded values
'''
import numpy as np

# mrh imports
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

# PySCF imports
from pyscf import gto, mcscf, scf
from qiskit import assemble, transpile

#from qiskit import Aer
from qiskit_aer import AerSimulator

#from mrh.my_pyscf.mcscf.lasci import h1e_for_cas
# Qiskit imports
#from qiskit.utils import QuantumInstance
from qiskit_algorithms import NumPyEigensolver

from postlas.get_geom import get_geom
from postlas.get_hamiltonian import get_hamiltonian


def test_hf():
    '''Test the PySCF RHF'''
    xyz = get_geom('close')
    mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log',
        symmetry=False)

    # Do RHF
    mf = scf.RHF(mol).run()
    assert(np.allclose(mf.e_tot, -2.0272406157535965))

def test_las():
    '''Test the MRH LASSCF against hardcoded value'''
    xyz = get_geom('close')
    mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log',
        symmetry=False)

    # Do RHF
    mf = scf.RHF(mol).run()

    # Set up LASSCF
    las = LASSCF(mf, (2,2),(2,2), spin_sub=(1,1))
    frag_atom_list = ((0,1),(2,3))
    loc_mo_coeff = las.localize_init_guess(frag_atom_list, mf.mo_coeff)

    # Run LASSCF
    las.kernel(loc_mo_coeff)
    assert(np.allclose(las.e_tot, -2.102635582026016))

def test_las_cas():
    '''Test the 1-fragment MRH LASSCF against CASSCF value'''
    xyz = get_geom('close')
    mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log',
        symmetry=False)

    # Do RHF
    mf = scf.RHF(mol).run()

    # Set up LASSCF
    las = LASSCF(mf, (4,),(4,), spin_sub=(1,))
    frag_atom_list = ((0,1,2,3),)
    loc_mo_coeff = las.localize_init_guess(frag_atom_list, mf.mo_coeff)

    # Run LASSCF
    las.kernel(loc_mo_coeff)

    # Set up CASSCF
    cas = mcscf.CASSCF(mf, 4, 4)

    # Run CASSCF
    cas.kernel(loc_mo_coeff)
    assert(np.allclose(las.e_tot, cas.e_tot))

def test_frag_np_eig():
    '''Test the LAS fragment numpy eigensolver against hardcoded value'''
    xyz = get_geom('close')
    mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log',
        symmetry=False)

    # Do RHF
    mf = scf.RHF(mol).run()

    # Set up LASSCF
    las = LASSCF(mf, (2,2),(2,2), spin_sub=(1,1))
    frag_atom_list = ((0,1),(2,3))
    loc_mo_coeff = las.localize_init_guess(frag_atom_list, mf.mo_coeff)

    # Run LASSCF
    las.kernel(loc_mo_coeff)
    loc_mo_coeff = las.mo_coeff

    # Using the built-in LASCI functions h1e_for_cas, get_h2eff
    h1_las = las.h1e_for_cas()
    eri_las = las.get_h2eff(loc_mo_coeff)

    # Storing each fragment's h1 and h2 as a list
    h1_frag = []
    h2_frag = []

    # Then construct h1, h2 for each fragment
    for idx in range(len(las.ncas_sub)):
        h1_frag.append(h1_las[idx][0][0])
        h2_frag.append(las.get_h2eff_slice(eri_las, idx))

    np_val = [-1.62002897,-1.59428151]
    for frag in range(len(las.ncas_sub)):
        hamiltonian = get_hamiltonian(frag, las.nelecas_sub, las.ncas_sub, h1_frag, h2_frag)

        # Set the backend
        backend = AerSimulator()
        qc = transpile(qc, backend)
        qobj = assemble(qc, shots=1024)
        #quantum_instance = QuantumInstance(backend = Aer.get_backend('aer_simulator'), shots=1024)

        # Numpy solver to estimate error in QPE energy due to trotterization
        np_solver = NumPyEigensolver(k=1)
        ed_result = np_solver.compute_eigenvalues(hamiltonian)
        print("NumPy result: ", ed_result.eigenvalues)
        assert(np.allclose(ed_result.eigenvalues[0], np_val[frag]))
