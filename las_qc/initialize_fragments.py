import numpy as np
from qiskit import QuantumRegister, QuantumCircuit

'''
This is for returning a quantum circuit. 
Four methods of initializing fragments:
    1. DI
    2. SF
    3. QPE
    4. VQE
'''

def get_soci_vec(ci_vec, nso, nelec):
        '''This is for DI'''
        lookup_a = {}
        lookup_b = {}
        cnt = 0
        norbs = nso // 2

        for ii in range(2**norbs):
            if f"{ii:0{norbs}b}".count("1") == nelec[0]:
                lookup_a[f"{ii:0{norbs}b}"] = cnt
                cnt += 1
        cnt = 0
        for ii in range(2**norbs):
            if f"{ii:0{norbs}b}".count("1") == nelec[1]:
                lookup_b[f"{ii:0{norbs}b}"] = cnt
                cnt += 1

        soci_vec = np.zeros(2**nso)
        for kk in range(2**nso):
            if (
                f"{kk:0{nso}b}"[norbs:].count("1") == nelec[0]
                and f"{kk:0{nso}b}"[:norbs].count("1") == nelec[1]
            ):
                so_ci_vec[kk] = ci_vec[
                    lookup_a[f"{kk:0{nso}b}"[norbs:]],
                    lookup_b[f"{kk:0{nso}b}"[:norbs]],
                ]

        return soci_vec

def direct_initialization(las):
    ncas = np.sum(las.ncas_sub)
    nqubits = 2*ncas
    qubits = np.arange(nqubits).tolist()
    frag_qubits = []
    i = 0
    for f in las.ncas_sub:
        frag_qubits.append(qubits[i : i + f] + qubits[ncas + i : ncas + i + f])
        i += f

    qr = QuantumRegister(nqubits) #SV (nqubits, 'lasq'); not sure if a name is needed
    circuit = QuantumCircuit(qr)

    for frag in range(len(las.ncas_sub)):
            circuit.initialize(get_soci_vec(las.ci[frag][0], 2*las.ncas_sub[frag], las.nelecas_sub[frag]), frag_qubits[frag])

    return circuit


def spectral_filtering():
    ...

def qpe_initialization():
    ...

def vqe_initialization():
    ...
