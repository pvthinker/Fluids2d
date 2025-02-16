import numpy as np
from collections import namedtuple
from .elliptic import *


class Mesh:
    """Mesh: class to handle Cartesian meshes

    dx and dy are uniform, though not necessarily equal

    if the default `msk` is changed then use `mesh.finalize()` to
    update all the derived informations: mskx, msky, mskv and the
    stencils

    """

    def __init__(self, param):
        self.param = param

        self.shape = get_shape(param)
        self.nx = param.nx
        self.ny = param.ny
        self.dx = param.Lx/self.nx
        self.dy = param.Ly/self.ny
        self.area = self.dx*self.dy

        self.xshift = 1
        self.yshift = self.shape[-1]

        self.set_default_mask()
        self.finalize()

    def finalize(self):
        self.set_masks()
        self.set_stencils(self.param.maxorder)

        self.poisson_centers = Poisson2D(self, "c")
        self.poisson_vertices = Poisson2D(self, "v")

        if self.param.model in ["qg", "qgrsw", "rsw"]:
            f0 = self.param.f0
            H = self.param.H
            g = self.param.g
            maindiag = self.area*f0**2/(g*H)
            self.qg_helmholtz = Poisson2D(self, "v", maindiag=maindiag)
            self.hb = 0
            self.qgcoef = f0/H

        if self.param.model in ["hydrostatic"]:
            self.poisson1d = Poisson1d(self)

    def _allocate(self):
        return np.zeros(self.shape, dtype="i1")

    def x(self, which):
        idx = get_idx(self.param, self.param.xperiodic, self.nx)
        shift = 0.5 if which in ["c", "y"] else 0.0
        return (idx+shift)*self.dx

    def y(self, which):
        idx = get_idx(self.param, self.param.yperiodic, self.ny)
        shift = 0.5 if which in ["c", "x"] else 0.0
        return (idx+shift)*self.dy

    def xy(self, which="c"):
        return np.meshgrid(self.x(which), self.y(which))

    def set_default_mask(self):
        self.msk = self._allocate()
        self.msk[:-1, :-1] = 1

    def set_masks(self):
        self.mskx = self._allocate()
        self.mskx[:, 1:] = self.msk[:, 1:]*self.msk[:, :-1]

        self.msky = self._allocate()
        self.msky[1:, :] = self.msk[1:, :]*self.msk[:-1, :]

        self.mskv = self._allocate()
        self.mskv[1:, 1:] = (self.msk[:-1, 1:]*self.msk[:-1, :-1]
                             * self.msk[1:, 1:]*self.msk[1:, :-1])

    def set_stencils(self, maxorder):
        Stencil = namedtuple("stencil", ("x", "y"))
        self.oc = Stencil(self._allocate(), self._allocate())
        self.ov = Stencil(self._allocate(), self._allocate())
        self.ok = Stencil(self._allocate(), self._allocate())

        set_order(self.msk, self.xshift, self.oc.x, maxorder)
        set_order(self.msk, self.yshift, self.oc.y, maxorder)

        set_order(self.mskv, -self.xshift, self.ov.x, maxorder)
        set_order(self.mskv, -self.yshift, self.ov.y, maxorder)

        set_order(self.mskx, -self.xshift, self.ok.x, maxorder)
        set_order(self.msky, -self.yshift, self.ok.y, maxorder)

    def fill(self, variable):
        if hasattr(variable, "_fields"):
            for k in range(len(variable)):
                self.fill(variable[k])
        else:
            fill_halo_array(self.param, variable)


def get_idx(param, periodic, n):
    if periodic:
        return np.arange(n+2*param.halowidth)-param.halowidth
    else:
        return np.arange(n+1)


def get_shape(param):
    n1 = (param.nx+2*param.halowidth
          if param.xperiodic else
          param.nx+1)
    n2 = (param.ny+2*param.halowidth
          if param.yperiodic else
          param.ny+1)

    return (n2, n1)


def fill_halo_array(param, array):
    if param.xperiodic:
        n = param.halowidth
        if array.ndim == 2:
            array[:, :n] = array[:, -2*n:-n]
            array[:, -n:] = array[:, n:2*n]
        elif array.ndim == 1:
            array[:n] = array[-2*n:-n]
            array[-n:] = array[n:2*n]


def set_order(msk, shift, stencil, maxorder):
    o = stencil.reshape(-1)
    m = msk.reshape(-1)
    n = len(o)

    if shift > 0:
        for i in range(n):
            if (i-shift >= 0):
                s2 = m[i-shift]+m[i]
            else:
                s2 = 0
            if (i-shift*2 >= 0) & (i+shift < n):
                s4 = m[i-shift*2]+m[i+shift] + s2
            else:
                s4 = 0
            if (i-shift*3 >= 0) & (i+2*shift < n):
                s6 = m[i-shift*3]+m[i+shift*2] + s4
            else:
                s6 = 0
            o[i] = (6 if s6 == 6 else
                    (4 if s4 == 4 else
                     (2 if s2 == 2 else 0)))
            o[i] = min(o[i], maxorder)
    else:
        for i in range(n):
            if (i-shift < n):
                s2 = m[i-shift]+m[i]
            else:
                s2 = 0
            if (i-shift*2 < n) & (i+shift >= 0):
                s4 = m[i-shift*2]+m[i+shift] + s2
            else:
                s4 = 0
            if (i-shift*3 < n) & (i+2*shift >= 0):
                s6 = m[i-shift*3]+m[i+shift*2] + s4
            else:
                s6 = 0
            o[i] = (6 if s6 == 6 else
                    (4 if s4 == 4 else
                     (2 if s2 == 2 else 0)))
            o[i] = min(o[i], maxorder)
