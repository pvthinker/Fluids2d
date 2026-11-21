import fluids2d as f2d
import numpy as np
from fluids2d.operators import compute_pv, qg_projection


def gaussian(x, y, x0, y0, r): return np.exp(-((x-x0)**2+(y-y0)**2)/(2*r**2))


def step(x, y, x0, y0, r): return (((x-x0)**2+(y-y0)**2) < r**2)*1.0


def set_initial(model, amp, r0=0.05, flow="dipole"):
    mesh = model.mesh
    x, y = mesh.xy("c")
    x0, y0, d = 0.5, 0.3, r0
    h = model.state.h

    H = model.param.H
    hb = mesh.hb

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
    h -= hb
    s = model.state

    if model.param.model == "qgrsw":
        qg_projection(mesh, s.u, s.h, s.pv, s.psi)

    elif model.param.model == "qg":
        qg_projection(mesh, s.U, s.h, s.pv, s.psi)
        pvback = mesh.hb*mesh.qgcoef
        s.pv[:, :] += pvback

    # elif model.param.model == "rsw":
    #     qg_projection(mesh, s.u, s.h, s.pv, s.p)

    model.integrator.diag(model.state)


def get_pv_diag():
    def diag_pv(param, mesh, s, time):
        compute_pv(param, mesh, s.omega, s.h, s.pv)
    return diag_pv


def set_experiment(name):
    param = f2d.Param()

    param.model = name
    param.nx = 100
    param.ny = 100
    param.tend = 100
    param.maxite = 5_000
    param.dtmax = 1
    param.f0 = 10.0

    param.animation = True
    param.nplot = 10
    param.plotvar = "pv"
    param.clims = [8, 12]

    param.nhis = 10
    if param.model in ["rsw", "qgrsw"]:
        param.var_to_store = ["h", "omega", "pv", "U"]

    elif param.model == "qg":
        param.plotvar = "pv"
        param.var_to_store = ["psi", "pv", "h"]

    if param.model == "rsw":
        timefactor = 25
        param.maxite *= timefactor
        param.nplot *= timefactor
        param.nhis *= timefactor

    amp = 0.2

    model = f2d.Model(param)

    def set_topo(mesh):
        x, y = mesh.xy()
        hb = 0.2*gaussian(x, y, 0.3, 0.7, 0.05)
        mesh.hb = hb*mesh.area*mesh.msk

    set_topo(model.mesh)

    set_initial(model, amp, r0=0.08, flow="dipole")

    if param.model != "qg":
        model.callbacks += [get_pv_diag()]

    return model


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.ion()

    model1 = set_experiment("qgrsw")
    # model2 = set_experiment("qg")

    # model1.set_dt()
    # model2.time.dt = model1.time.dt

    # f2d.tools.run_twin_experiments(model1, model2)
    model1.run()
