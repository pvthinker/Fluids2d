# Fluids2d

[![PyPI - Version](https://img.shields.io/pypi/v/fluids2d.svg)](https://pypi.org/project/fluids2d)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fluids2d.svg)](https://pypi.org/project/fluids2d)

-----

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [First experiment](#first-experiment)
- [License](#license)


## Description

`Fluids2d` is a versatile Python CFD code that solves a large
class of 2D flows. It is designed to be used easily by Students
learning Fluid Mechanics or Geophysical Fluid Dynamics and by their
professors willing to illustrate their course and to organize
numerical practicals. The idea is to visualize flows on the fly, as
they are computed. The effect of parameter changes can be seen
immediately.

`Fluids2d` is based on `Fluid2d` (without "s"), with an easier
install, a better implementation, a better numerics and soon more
equations. It already includes the RSW equations and the QG equations
solved with a projection method (Thiry et al 2024). The code is fast,
thanks to the Python modules `numba` and `scipy.sparse.linalg`.


You can learn how basic processes work because of the power of
animations. It is quite easy to go beyond textbooks and to reach
research questions. But remember, the code is limited to 2D flows,
this is a serious limitation, especially for flows in the vertical
plane. The flows have a tendency to develop an inverse cascade of
energy, viz. vortices get bigger with time, which is unrealistic.

The code is also useful to learn about numerical methods.

Several features are particulary cool

 - the code handles many different sets of equations: transport of
   scalars and of vectors, Euler, Boussinesq, Rotating Shallow Water

 - the code handles a mask system that allows to have convoluted
   geometries (closed domain with arbitratry shape, reentrant channel,
   with multiple island etc)

 - the code tends to have a very low level of dissipation because the
   dissipation is handled implicitely by the numerics thanks to WENO
   reconstructions.

 - the code can integrate in parallel two models, that differ by the
   physics or by the numerics.


## Installation

- Create a new virtual environnement, for instance
```console
conda create -n fluids2d
```
- activate it
```console
conda activate fluids2d
```
- Go into `fluids2d` main folder
```console
python -m pip install -e .
```
## First experiment

Copy the folder of experiments and work in it
```console
cd fluids2d
cp -pR src/experiments  myexp
cd myexp
python vortex.py
```
the results are saved in the NetCDF file `history.nc`.

## License

`Fluids2d` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
