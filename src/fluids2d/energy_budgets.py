import numpy as np
import xarray as xr
from .restart_tools import load_from_restart
from .dissipation_diag import Dissipation, crop

class EnergyBudget:
    def __init__(self, model, nt=1000):
        self.dissip = Dissipation(model)

        if hasattr(model, "Qforc"):
            self.Qforc = model.Qforc
        else:
            self.Qforc = model.state.b*0

        self.allocate(nt)
        x, self.z = model.mesh.xy("cc")
        self.state = model.state
        self.dx = model.mesh.dx
        self.dy = model.mesh.dy
        self.nh = model.param.halowidth

        self.msk = model.mesh.msk
        self.ncells = np.sum(self.msk)

    def allocate(self, nt):
        names = ["epsK", "epsP", "epsB", "epsb2",
                 "BPE", "APE", "KE", "ForcA", "Conv"]
        self.data={name: np.zeros(nt) for name in names}
        self.kt = 0

    def update_timeseries(self,data):
        for k,v in data.items():
            self.data[k][self.kt] = v
        self.kt += 1

    def domain_average(self, x):
        """ domain average of array x"""
        return np.sum(x*self.msk)/self.ncells

    def compute(self, *args):

        mean = lambda x: self.domain_average(x)

        nh = self.nh
        self.dissip.compute()
        b = self.state.b
        u = self.state.u.x
        v = self.state.u.y
        d = self.dissip
        bp = self.dissip.bp
        bpe = -bp.bval*bp.zval
        ape = np.zeros_like(b)
        ape[nh:-nh,nh:-nh] = bp.ape(crop(b), bp.z)
        ke = 0.5*(mean(u**2)/self.dx**2+mean(v**2)/self.dy**2)
        zrb = bp.zr(crop(b))
        forcA = np.zeros_like(b)
        forcA[nh:-nh,nh:-nh] = crop(self.Qforc)*(zrb-bp.z)

        conv = np.zeros_like(b)
        conv[nh:-nh,nh:-nh] =  crop(b)-bp.br(bp.z)
        conv[:-1,:] *= 0.5*(v[1:,:]+v[:-1,:])/self.dy


        self.update_timeseries({"epsK": mean(d.epsK),
                                "epsP": mean(d.epsP),
                                "epsB": mean(d.epsB),
                                "epsb2": mean(d.epsb2),
                                "BPE": np.mean(bpe),
                                "APE": mean(ape),
                                "KE": ke,
                                "ForcA": mean(forcA),
                                "Conv": mean(conv),
                                })

def compute_budgets_from_netcdf(model, ncfile, nt=-1):
    ds = xr.load_dataset(ncfile)

    if nt==-1:
        nt = len(ds.t)

    budget = EnergyBudget(model, nt)

    for k in range(nt):
        load_from_restart(model,ncfile,k)
        budget.compute()
        print(f"\rprocess ite={k}/{nt}", end="")

    budget.data["time"] = ds.t.data[:nt]

    return budget
