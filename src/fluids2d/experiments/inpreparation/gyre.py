import numpy as np
import fluids2d as f2d


def set_forcing(model, tau0=0.1):
    mesh = model.mesh
    x, y = mesh.xy()
    Ly = model.param.Ly

    ky = np.pi/Ly
    x, y = mesh.xy("v")
    tau = -tau0*np.sin(ky*y)
    tau *= mesh.mskx * mesh.dx
    curltau = -tau0*ky*np.cos(ky*y)*mesh.msk*mesh.area

    def forcing(param, mesh, s, ds):
        ds.pv[:, :] += curltau

    model.add_forcing(forcing)


def set_initialstate(model):

    param = model.param

    def f(y): return param.beta*(y-param.Ly/2)*model.mesh.area

    x, y = model.mesh.xy("v")
    model.mesh.f = f(y)

    x, y = model.mesh.xy()
    pv = model.state.pv
    pv += f(y)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    plt.ion()

    param = f2d.Param()

    param.model = "qg"
    param.Lx = 2
    param.ny = 50*2
    param.nx = param.ny*param.Lx
    param.tend = 600
    param.dtmax = 0.1
    param.maxite = 20_000
    param.xperiodic = False
    param.f0 = 25
    param.beta = 10

    param.nplot = 20
    param.animation = True
    param.plotvar = "pv"

    param.nhis = 20
    param.var_to_store = ["pv", "psi"]

    model = f2d.Model(param)
    set_initialstate(model)
    set_forcing(model, tau0=5e-3)

    model.run()
