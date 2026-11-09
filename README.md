
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

By incorporating dynamical optimal transport, the resulting interpolation can closely approximate the Wasserstein-2 geodesic between the given copulas.

## Main Features

- Continuous interpolation between checkerboard copulas.
- Preservation of uniform marginals along the interpolation path.
- Construction using adjacent-cell mass movements.
- Displacement-like interpolation guided by dynamical optimal transport.
- Implementations for two- and three-dimensional copulas.

## Numerical Examples

The accompanying paper considers the following examples:

### 1. Gaussian and Gumbel Copulas

Interpolation between a Gaussian copula with correlation parameter 0.8 and a Gumbel copula with parameter 2.3.

The experiment illustrates the evolution of the dependence structure and compares the resulting interpolation with the Wasserstein displacement path.

### 2. Fuel Injector Data

An application to dependence modeling under different operating conditions of a jet-engine fuel injector.

The interpolation models the evolution of dependence between droplet position and velocity.

### 3. Three-Dimensional Copulas

Interpolation between checkerboard approximations of two three-dimensional Gaussian copulas, illustrating the extension to higher dimensions.
