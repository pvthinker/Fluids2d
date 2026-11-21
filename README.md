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

`Fluids2d` requires **Python ≥ 3.9** (recommended: 3.10–3.12), `git` and
`pip`. The package is installed in *editable* mode (`pip install -e .`),
which means that the code in `src/fluids2d` is used directly from your
clone: any modification is effective at the next run, without reinstalling —
ideal for practicals. This single command also installs all the dependencies
(`numpy`, `scipy`, `matplotlib`, `numba`, `netCDF4`, `xarray`, `pillow`).

### Quick start (venv + pip, recommended)

```console
git clone https://github.com/pvthinker/fluids2d.git
cd fluids2d
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e .
```

### Alternative: conda

If you prefer `conda` (convenient to get fast binary packages of `numba`
and `netCDF4`):

```console
conda create -n fluids2d python=3.12
conda activate fluids2d
python -m pip install -e .
```

### Check the installation

```console
python -c "from fluids2d.__about__ import __version__; print('fluids2d', __version__)"
python -m unittest discover -s tests
```

### Optional extras

- **MP4 animations** — set `param.generate_mp4 = True` to export movies;
  this requires the `ffmpeg` (or `avconv`) executable:
  `conda install -c conda-forge ffmpeg`, `brew install ffmpeg` (macOS) or
  `sudo apt install ffmpeg` (Debian/Ubuntu).
- **Headless machines** (remote server without a screen): use the
  non-interactive matplotlib backend with `export MPLBACKEND=Agg`.
- **Developer tools** (ruff linter + pre-commit hooks):
  `python -m pip install -e .[dev]`.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: fluids2d` | activate your environment again, or run `python -m pip install -e .` from the `fluids2d` main folder |
| `ModuleNotFoundError: <package>` | run `python -m pip install <package>` |
| old tutorials use `import experiments` | the experiments now live in `fluids2d.experiments`; use `from fluids2d.experiments import vortex` |
| `matplotlib` fails on a remote server | `export MPLBACKEND=Agg` |

## First experiment

Copy the folder of experiments and work in it
```console
cd fluids2d
cp -pR src/fluids2d/experiments  myexp
cd myexp
python vortex.py
```
the results are saved in the NetCDF file `history.nc`.

## License

`Fluids2d` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
