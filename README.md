# LAS + Quantum Computing

This repository will eventually contain a package that will be able to run the following algorithms:

1. LAS-UCC: Previous repo at https://github.com/GagliardiGroup/las-qpe
2. LAS-QKSD: Previous repo at https://github.com/GagliardiGroup/las-qksd
3. LAS-nuVQE: Previous repo at https://github.com/joannaqw/LAS-nuVQE
4. LAS-USCC: Previous repo at https://github.com/GagliardiGroup/LAS-USCC + polynomial algorithm incorporated into MRH
5. LAS-ADAPT:
6. LAS-SQD: 

## Requirements
The code requires:

1. Qiskit version 0.5.7
2. Qiskit-nature version 0.7.2
3. Qiskit-aer version 0.15.2
4. Qiskit-algorithms version 0.3.1
5. Qiskit-terra version 0.24.0
6. Qiskit-ibmq-provider 0.20.2
3. MRH
4. PySCF version 2.12.1

## Installation

This package depends on MRH which in turn requires a specific distribution of PySCF and PySCF forge. If you find that the installation fails, please ensure that the correct PySCF is installed.

<!-- TODO: Make instructions on how to check this. -->

```bash
# Install PySCF and PySCF-Forge. See the MRH reposity for the specific versions
pip install https://github.com/GagliardiGroup/las-qc.git
```

### Developer Installation

Clone this repository and perform an editable install with pip.

```bash
git clone https://github.com/GagliardiGroup/las-qc.git
pip install -e ./las-qc[dev]
```

## Running

The CLI hasn't been started, but for now you can run a sample calculation by installing LAS-QC and running

```bash
las-qc
```
