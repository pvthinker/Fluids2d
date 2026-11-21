import fluids2d as f2d
import numpy as np
from fluids2d.operators import compute_pv, qg_projection


def gaussian(x, y, x0, y0, r): return np.exp(-((x-x0)**2+(y-y0)**2)/(2*r**2))


def step(x, y, x0, y0, r): return (((x-x0)**2+(y-y0)**2) < r**2)*1.0


def set_initial(model, amp, r0=0.05, flow="dipole"):
    mesh = model.mesh
    x, y = mesh.xy("c")
    x0, y0, d = 0.5, 0.5, r0
    h = model.state.h

    H = model.param.H

    if flow == "vortex":
        h[:] = H + amp*gaussian(x, y, x0, y0, r0)

    elif flow == "dipole":
        h[:] = H + amp*(gaussian(x, y, x0+d, y0, r0) -
                        gaussian(x, y, x0-d, y0, r0))

    elif flow == "bumpedvortex":
        h[:] = H + amp*np.maximum(step(x, y, x0, y0, r0),
                                  step(x, y, x0+r0, y0, r0/2))

    elif flow == "dambreak":
        h[:] = H + amp*np.tanh((y-y0)/0.02)

    else:
        raise NotImplemented

    h *= (mesh.msk*mesh.area)
    s = model.state

    if model.param.model == "qgrsw":
        qg_projection(mesh, s.u, s.h, s.pv, s.psi)

    elif model.param.model == "qg":
        qg_projection(mesh, s.U, s.h, s.pv, s.psi)

    # elif model.param.model == "rsw":
    #     qg_projection(mesh, s.u, s.h, s.pv, s.p)

    model.integrator.diag(model.state)


def get_pv_diag():
    def diag_pv(param, mesh, s, time):
        compute_pv(param, mesh, s.omega, s.h, s.pv)
    return diag_pv


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.ion()

    param = f2d.Param()

    param.model = "rsw"
    param.nx = 100
    param.ny = 100
    param.tend = 400
    param.maxite = 500
    param.dtmax = 1
    param.f0 = 10.0

    param.animation = True
    param.nplot = 5
    param.plotvar = "h"
    param.xperiodic = False

    param.integrator = "rk3"

    if param.integrator == "LFRA":
        param.cfl = 0.5
        param.compflux = "centered"
        param.vortexforce = "centered"
        param.innerproduct = "classic"

    param.nhis = 10
    if param.model == "rsw":
        param.var_to_store = ["h", "omega", "pv"]
    else:
        param.var_to_store = ["psi", "pv"]

    amp = 0.2

    model = f2d.Model(param)
    set_initial(model, amp, r0=0.1, flow="dipole")

    if param.model != "qg":
        model.callbacks += [get_pv_diag()]

    model.run()
