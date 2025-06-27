#########################
# Script to run LASSCF, use the statevectors from the fragments  
# to run a VQE using a generalized UCCSD ansatz on the whole system
# This script either uses saved QPE statevectors or
# a LASCI vector initialized on the qubits
# It can also use an HF initial state
#########################

import logging
import time
from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np
from custom_UCC import custom_UCC

#from c4h6_struct import structure
from get_geom import get_geom
from mrh.exploratory.unitary_cc import lasuccsd

# mrh imports
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

# PySCF imports
from pyscf import ao2mo, gto, lib, mcscf, scf
from qiskit import Aer, QuantumCircuit, QuantumRegister, transpile
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit.utils import QuantumInstance

# Qiskit imports
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper

parser = ArgumentParser(description='Do LAS-VQE, specifying num of ancillas and shots')
parser.add_argument('--dist', type=float, default=1.35296239, help='distance of H2s from one another or C=C bond distance scaling in butadiene')
parser.add_argument('--shots', type=int, default=1024, help='number of shots for the simulator')
args = parser.parse_args()

# Prints info at every VQE iteration
logging.basicConfig(level='DEBUG')

# Define molecule: (H2)_2
xyz = get_geom('scan', dist=args.dist)
#xyz = '''H 0.0 0.0 0.0
#             H 1.0 0.0 0.0
#             H 0.2 1.6 0.1
#             H 1.159166 1.3 -0.1'''
mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g_{}.log'.format(args.dist),
    symmetry=False, verbose=lib.logger.DEBUG)

# Do RHF
mf = scf.RHF(mol).run()
print("HF energy: ", mf.e_tot)

'''
# Define molecule: C_4H_6
norb = 8
nelec = 8
norb_f = (4,4)
nelec_f = ((2,2),(2,2))
mol = structure (0.0, 0.0, output='c4h6_631g_{}.log'.format(args.dist), verbose=lib.logger.DEBUG)

# Do RHF
mf = scf.RHF(mol).run()
print("HF energy: ", mf.e_tot)

#CASSCF (for comparison with Matt's code)
mc = mcscf.CASSCF (mf, norb, nelec).set (fcisolver = csf_solver (mol, smult=1))
mo_coeff = mc.sort_mo ([11,12,14,15,16,17,21,24])
mc.kernel (mo_coeff)
mo_coeff = mc.mo_coeff.copy ()

# Create LASSCF object
las = LASSCF (mf, norb_f, nelec_f, spin_sub=(1,1)).set (mo_coeff=mo_coeff)

# Localize the chosen fragment active spaces
loc_mo_coeff = las.localize_init_guess ([[0,1,2,3,4],[5,6,7,8,9]])
las.kernel(loc_mo_coeff)
loc_mo_coeff = las.mo_coeff
print("LASSCF energy: ", las.e_tot)
'''
# Create LASSCF object
# Keywords: (wavefunction obj, num_orb in each subspace, (nelec in each subspace)/((num_alpha, num_beta) in each subspace), spin multiplicity in each subspace)
las = LASSCF(mf, (2,2),(2,2), spin_sub=(1,1))
#las = LASSCF(mf, (4,),(4,), spin_sub=(1,))

# Localize the chosen fragment active spaces
frag_atom_list = ((0,1),(2,3))
#frag_atom_list = ((0,1,2,3),)
loc_mo_coeff = las.localize_init_guess(frag_atom_list, mf.mo_coeff)

# Run LASSCF
las.kernel(loc_mo_coeff)
loc_mo_coeff = las.mo_coeff
print("LASSCF energy: ", las.e_tot)

ncore = las.ncore
ncas = las.ncas
ncas_sub = las.ncas_sub
nelec_cas = las.nelecas

# Checking subspace sizes and num. of electrons
print ("Ncore: ", ncore, "Ncas: ", ncas, "Ncas_sub: ", ncas_sub, "Nelec_cas: ", nelec_cas)

# CASCI h1 & h2 for VQE Hamiltonian
mc = mcscf.CASCI(mf,4,4)
mc.kernel(loc_mo_coeff)
cas_h1e, e_core = mc.h1e_for_cas()

eri_cas = mc.get_h2eff(loc_mo_coeff)
eri = ao2mo.restore(1, eri_cas,mc.ncas)

# Gets all acceptable operators for UCCSD
# excluding intra-fragment ones
def custom_excitations(num_spin_orbitals: int,
                           num_particles: Tuple[int, int],
                           num_sub: List[int]
                           ) -> List[Tuple[Tuple[int, ...], ...]]:
    excitations = []
    norb = int(num_spin_orbitals/2)
    uop = lasuccsd.gen_uccsd_op (norb, num_sub)
    a_idxs = uop.a_idxs
    i_idxs = uop.i_idxs
    for a,i in zip(a_idxs,i_idxs):
        excitations.append((tuple(i),tuple(a[::-1])))
    
    return excitations

raise NotImplementedError("get_hamiltonian is not implemented yet!")
# hamiltonian = get_hamiltonian(None, mc.nelecas, mc.ncas, cas_h1e, eri)
hamiltonian = None
#print(hamiltonian)

# Loading the QPE statevector from file
state_list = np.load('qpe_state_{}.npy'.format(args.dist), allow_pickle=True)

# Create a quantum register with system qubits
qr1 = QuantumRegister(np.sum(ncas_sub)*2, 'q1')
new_circuit = QuantumCircuit(qr1)
new_circuit.initialize(state_list[0], qubits=[0,1,4,5])
new_circuit.initialize(state_list[1], qubits=[2,3,6,7])

# Gate counts for initialization
if args.dist == 0.0:
    target_basis = ['rx', 'ry', 'rz', 'h', 'cx']
    circ_for_counts = transpile(new_circuit, basis_gates=target_basis, optimization_level=0)
    init_op_dict = circ_for_counts.count_ops()
    init_ops = sum(init_op_dict.values())
    print("Operations: {}".format(init_op_dict))
    print("Total operations: {}".format(init_ops))

# Tracking the convergence of the VQE
counts = []
values = []
def store_intermediate_result(eval_count, parameters, mean, std):
    counts.append(eval_count)
    values.append(mean)

qubit_converter = QubitConverter(mapper = JordanWignerMapper(), two_qubit_reduction=False)
new_instance = QuantumInstance(backend = Aer.get_backend('aer_simulator'), shots=args.shots)
# Setting up the VQE
ansatz = custom_UCC(qubit_converter=qubit_converter, num_particles=(2,2), num_spin_orbitals=8, excitations=custom_excitations, initial_state=new_circuit, preserve_spin=False)
optimizer = L_BFGS_B(maxfun=10000, iprint=101)
init_pt = np.zeros(146)
#optimizer = COBYLA(maxiter=1000)
algorithm = VQE(ansatz=ansatz, optimizer=optimizer, quantum_instance=new_instance, initial_point=init_pt, callback=store_intermediate_result) 

# Gate counts for VQE (includes initialization)
if args.dist == 0.0:
    params = np.zeros(146)
    vqe_ops = 0
    circ_list = transpile(algorithm.construct_circuit(params, hamiltonian), basis_gates=target_basis, optimization_level=0)
    for circ in circ_list:
        vqe_op_dict = circ.count_ops()
        vqe_ops += sum(vqe_op_dict.values())
    print("Number of circuits in list: ",len(circ_list))
    print("Operations: {}".format(vqe_op_dict))
    print("Total operations: {}".format(vqe_ops))

# Running the VQE
t0 = time.time()
vqe_result = algorithm.compute_minimum_eigenvalue(hamiltonian)
print(vqe_result)
t1 = time.time()
print("Time taken for VQE: ",t1-t0)
print("VQE counts: ", counts)
print("VQE energies: ", values)

# Saving all relevant results in a dict
if args.dist == 0.0:
    np.save('results_{}_{}_vqe.npy'.format(args.shots, args.dist), {'init_op_dict': init_op_dict, 'init_ops': init_ops, 'vqe_op_dict':vqe_op_dict, 'vqe_ops': vqe_ops, 'vqe_result':vqe_result, 'vqe_en_vals':values, 'vqe_counts':counts, 'nuc_rep': las.energy_nuc()})
else:
    np.save('results_{}_{}_vqe.npy'.format(args.shots, args.dist), {'vqe_result':vqe_result, 'vqe_en_vals':values, 'vqe_counts':counts, 'nuc_rep': las.energy_nuc()})
