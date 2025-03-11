from netCDF4 import Dataset
from collections import namedtuple
import numpy as np
from .io import get_atts_from_param
import os


class Bulk:
    def __init__(self, model):
        self.model = model

        nh = model.param.halowidth
        def domsum(x): return np.sum(x, axis=None)
        def domavg(x): return domsum(x)/domsum(model.mesh.msk)
        self.domavg = domavg

        variables = ("time", "ke", "ens", "vort", "angular")
        self.Bulk = namedtuple("bulk", variables)

        self.ndiags = 1_000
        self.data = self.Bulk(*[np.zeros((self.ndiags,))
                                for _ in variables])

        xv, yv = model.mesh.xy("y")
        xu, yu = model.mesh.xy("x")
        self.xvyu = (xv, yu)

        self.kt = 0

        self.ncfile = "bulk.nc"
        self.k0 = get_number_of_records(self.ncfile)
        if self.k0 == 0:
            self.create_newfile()

    def create_newfile(self):
        create_file(self.ncfile, self.model.param, self.data._fields)
        self.k0 = 0

    def __call__(self):
        if self.model.time.ite % 3 > 0:
            return

        mesh = self.model.mesh
        kt = self.kt
        s = self.model.state
        xv, yu = self.xvyu
        dx, dy = mesh.dx, mesh.dy
        area = mesh.area

        time, ke, ens, vort, angular = self.data

        ke[kt] = self.domavg(s.ke)
        ens[kt] = 0.5*self.domavg(s.omega**2)/area**2
        vort[kt] = self.domavg(s.omega)/area
        angular[kt] = (self.domavg(s.U.y*xv)*dx
                       - self.domavg(s.U.x*yu)*dy)

        time[kt] = self.model.time.t

        self.kt += 1
        if self.kt == self.ndiags:
            self.write()

    def finalize(self):
        self.write()

    def write(self):
        n = self.kt
        idx = slice(self.k0, self.k0+n)
        with Dataset(self.ncfile, "r+") as nc:

            for name, x in zip(self.data._fields, self.data):
                nc.variables[name][idx] = x[:n]

        self.k0 += n
        self.kt = 0

    def read(self):
        with Dataset(self.ncfile, "r") as nc:
            data = self.Bulk(*[nc.variables[name][:]
                               for name in self.Bulk._fields])
        return data


def create_file(ncfile, param, variables):
    dtype = "f"

    atts = get_atts_from_param(param)

    with Dataset(ncfile, "w", format='NETCDF4') as nc:
        nc.setncatts(atts)

        nc.createDimension("t", None)

        for name in variables:
            v = nc.createVariable(name, dtype, ("t", ))
            v.standard_name = name


def get_number_of_records(ncfile):
    if os.path.isfile(ncfile):
        with Dataset(ncfile) as nc:
            nrec = len(nc.dimensions["t"])
    else:
        nrec = 0
    return nrec
