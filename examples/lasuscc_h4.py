import numpy as np
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from las_qc.lasucc import LASUCC

# Initializing the molecule with RHF
#===================================
xyz = '''H 0.000000 0.000000 0.000000
         H 1.889726 0.000000 0.000000
         H 0.377945 3.023562 0.188973
         H 2.190506 2.456644 -0.188973'''
mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log.py',
    verbose=4)
mf = scf.RHF (mol).run ()
ref = mcscf.CASSCF (mf, 4, (3,1)).run (verbose=4) # = FCI
print ("ref CAS = ", ref.e_tot)

# Running LASSCF
#===================================
las = LASSCF (mf, (2,2), ((2,0),(1,1)), spin_sub=(3,1))
las.verbose=4
frag_atom_list = ((0,1),(2,3))
mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
las.kernel (mo_loc)
print ("LASSCF = ", las.e_tot)

# Running CASCI@LAS
#===================================
mc = mcscf.CASCI(mf, 4, (3,1)).run()
mc.mo_coeff = las.mo_coeff
print ("ref CASCI@LAS = ", mc.e_tot)

# LAS-USCC-VQE
#=====================================================
frag_orbs = (2,2)
frag_elecs = ((2,0),(1,1))
frag_spins = (3,1)

eps = [0.01, 0.001]

from las_qc.lasqc import LASQC
solver = LASUCC(
    mol,
    las=las,
    frag_orbs=(2,2),
    frag_elec=((2,0),(1,1)),#(2,2),
    frag_atom_list=((0,1),(2,3)),
    spin_sub=(3,1),
    epsilon=0.01, # pass epsilon=0.0 for LAS-UCC
)
vqe_en = solver.run()
print(f"LAS-VQE Energy: {vqe_en:.12f} Ha | Epsilon: {eps[0]:.12f}")
