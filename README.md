# H₂ Orbital JAX

> GPU-accelerated construction and visualization of the H₂ molecular orbitals using the Linear Combination of Atomic Orbitals (LCAO), JAX, and Marching Cubes.

![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![JAX](https://img.shields.io/badge/JAX-GPU-orange)

---

## Overview

**H₂ Orbital JAX** is a computational physics project that reconstructs the bonding and antibonding molecular orbitals of the hydrogen molecule from first principles.

Instead of treating molecular orbitals as pre-rendered illustrations, this project evaluates the wavefunction on a three-dimensional Cartesian grid, reconstructs the scalar field, extracts isosurfaces using Marching Cubes, and exports the resulting geometry for visualization.

The implementation combines

- Quantum Mechanics
- Molecular Orbital Theory (LCAO)
- GPU Computing with JAX
- Scientific Visualization
- Computational Geometry

into a fully reproducible numerical workflow.

---

## Features

- GPU-accelerated computation with **JAX**
- Analytical LCAO construction of H₂ molecular orbitals
- Numerical normalization and overlap verification
- Orthogonality validation
- Probability-based isosurface extraction
- Relative-amplitude thresholding
- Marching Cubes mesh reconstruction
- Interactive Plotly visualization
- OBJ / PLY / GLB export
- Automatic experiment metadata generation

---

## Computational Pipeline

The complete numerical workflow is

```math
\psi(\mathbf r)
\longrightarrow
|\psi(\mathbf r)|^2
\longrightarrow
\text{3D Scalar Field}
\longrightarrow
\text{Isosurface}
\longrightarrow
\text{Triangle Mesh}
```

---

## Physical Model

Two hydrogen nuclei are positioned along the x-axis

```math
\mathbf R_A=
\left(-\frac R2,0,0\right),
\qquad
\mathbf R_B=
\left(+\frac R2,0,0\right).
```

The normalized hydrogenic 1s orbital is

```math
\phi_{1s}(r)
=
\frac{e^{-r}}
{\sqrt{\pi}}.
```

The analytical overlap integral is

```math
S(R)
=
e^{-R}
\left(
1+R+\frac{R^2}{3}
\right).
```

The bonding molecular orbital is

```math
\psi_\sigma
=
\frac{\phi_A+\phi_B}
{\sqrt{2(1+S)}}.
```

The antibonding molecular orbital is

```math
\psi_{\sigma^\ast}
=
\frac{\phi_A-\phi_B}
{\sqrt{2(1-S)}}.
```

---

## Numerical Validation

The implementation automatically verifies

- atomic orbital normalization
- molecular orbital normalization
- analytical overlap
- numerical overlap
- bonding–antibonding orthogonality

Typical output

```text
Analytic overlap S(R):              0.7529427409
Numerical overlap:                  0.7529422641
Absolute overlap error:             4.77e-07

Bonding orbital norm:               0.9999988
Antibonding orbital norm:           0.9999955

Bonding-antibonding overlap error:  1.10e-08
```

The numerical error remains on the order of floating-point precision.

---

## Isosurface Extraction

Two thresholding strategies are available.

### Probability Threshold

A density threshold

```math
\rho_c
```

is determined such that the enclosed region contains a prescribed fraction of the total probability

```math
\frac{
\int_{\rho\ge\rho_c}
\rho\,dV
}
{
\int
\rho\,dV
}
=
P.
```

The extracted surfaces satisfy

```math
\psi(\mathbf r)
=
\pm\sqrt{\rho_c}.
```

---

### Relative Amplitude

Alternatively,

```math
\psi
=
\pm
\eta
\max|\psi|
```

can be used.

---

## Gallery

### Bonding Molecular Orbital

![Bonding](img/2026-07-25%20223001.png)

---

### Antibonding Molecular Orbital

![Antibonding](img/2026-07-25%20223026.png)

---

## Project Structure

```text
Cell 1
Environment & Dependencies

Cell 2
Experiment Configuration

Cell 3
Physical Model

Cell 4
Numerical Utilities

Cell 5
Threshold Algorithms

Cell 6
Marching Cubes

Cell 7
Visualization

Cell 8
Pipeline

Cell 9
Run Experiment

Cell 10
Experiment Summary
```

---

## Output

The program exports

```text
experiment_metadata.json
```

which records

- simulation parameters
- software versions
- JAX device information
- overlap validation
- normalization errors
- orthogonality errors
- mesh statistics

making every generated model fully reproducible.

---

## Future Work

- H₂⁺ molecular ion
- Potential energy curve
- 2s / 2p orbitals
- Gaussian basis functions
- Hartree–Fock
- Volume rendering
- Blender integration
- Python package

---

## License

MIT License
