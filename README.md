
# Continuous Displacement Interpolation Between Checkerboard Copulas

This repository contains Python implementations of continuous displacement-like interpolation between checkerboard copulas in two and three dimensions.

The code accompanies the paper:

**Grothe, O., & Luo, N. (2026).**
[Continuous displacement interpolation between checkerboard copulas](https://doi.org/10.1515/demo-2025-0018).
*Dependence Modeling*, 14(1), 20250018.

## Overview

Copulas describe the dependence structure between random variables independently of their marginal distributions. Interpolating between two copulas provides a way to model how dependence structures evolve between different states.

A natural approach is displacement interpolation based on optimal transport. However, standard Wasserstein displacement interpolation does not generally preserve the uniform marginals required of a copula.

This project implements a displacement-like interpolation method for checkerboard copulas. The approach constructs continuous paths through elementary mass movements between neighboring cells, preserving the copula property throughout the interpolation.

To guide the interpolation, intermediate distributions along a dynamical optimal transport path are transformed into checkerboard copulas. These copulas are then connected through continuous local interpolations, yielding a displacement-like path that preserves uniform marginals.

## Numerical Examples

The accompanying paper considers the following examples:

### 1. Gaussian and Gumbel Copulas

Interpolation between checkerboard approximations of a Gaussian copula with correlation parameter 0.8 and a Gumbel copula with parameter 2.3.

The experiment illustrates the evolution of the dependence structure and compares the resulting interpolation with the Wasserstein displacement path.

### 2. Fuel Injector Data

An application to the dependence between droplet y-position and x-velocity under two operating conditions of a jet-engine fuel injector.

### 3. Three-Dimensional Gaussian Copulas

Interpolation between checkerboard approximations of two three-dimensional Gaussian copulas with different correlation matrices, illustrating the extension of the method to higher dimensions.

## Repository Contents

The main code and example notebooks are located in `interpolation_checkerboard_copula/`.

| File or directory | Description |
| --- | --- |
| `dim2_interpolation.ipynb` | Two-dimensional interpolation examples, including the fuel injector application. |
| `dim3_interpolation.ipynb` | Three-dimensional interpolation examples. |
| `dim2_OT_inter.py` | Numerical routines for dynamical optimal transport in two dimensions. |
| `dim3_OT_inter.py` | Numerical routines for dynamical optimal transport in three dimensions. |
| `02_Estimation_t2000test/` | Data files used in the fuel injector example. |

## Usage

1. Download or clone this repository.
2. Start Jupyter Notebook from the `interpolation_checkerboard_copula/` directory.
3. Open `dim2_interpolation.ipynb` for the two-dimensional examples, or `dim3_interpolation.ipynb` for the three-dimensional examples.
4. Run the cells for the selected example in order, with the required Python packages installed.

Keep the original directory structure, as the notebooks import the local Python modules and access data files using relative paths.
