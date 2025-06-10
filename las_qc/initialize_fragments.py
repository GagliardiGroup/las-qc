import numpy as np

'''
This is for returning a quantum circuit. 
Four methods of initializing fragments:
    1. DI
    2. SF
    3. QPE
    4. VQE
'''

def get_soci_vec(ci_vec, num_spin_orbitals, nelec):
        '''This is for DI'''
        lookup_a = {}
        lookup_b = {}
        cnt = 0
        norbs = num_spin_orbitals // 2

        for ii in range(2**norbs):
            if f"{ii:0{norbs}b}".count("1") == nelec[0]:
                lookup_a[f"{ii:0{norbs}b}"] = cnt
                cnt += 1
        cnt = 0
        for ii in range(2**norbs):
            if f"{ii:0{norbs}b}".count("1") == nelec[1]:
                lookup_b[f"{ii:0{norbs}b}"] = cnt
                cnt += 1

        soci_vec = np.zeros(2**num_spin_orbitals)
        for kk in range(2**num_spin_orbitals):
            if (
                f"{kk:0{num_spin_orbitals}b}"[norbs:].count("1") == nelec[0]
                and f"{kk:0{num_spin_orbitals}b}"[:norbs].count("1") == nelec[1]
            ):
                so_ci_vec[kk] = ci_vec[
                    lookup_a[f"{kk:0{num_spin_orbitals}b}"[norbs:]],
                    lookup_b[f"{kk:0{num_spin_orbitals}b}"[:norbs]],
                ]

        return soci_vec

def direct_initialization():
    ...

def spectral_filtering():
    ...

def qpe_initialization():
    ...

def vqe_initialization():
    ...
