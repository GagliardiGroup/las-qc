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

1. Qiskit version 1.x.x
2. Qiskit-nature version 0.x
3. MRH
4. PySCF version 2.x

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
