import numpy as np
import fluids2d as f2d


def gaussian(x, y, x0, y0, r): return np.exp(-((x-x0)**2+(y-y0)**2)/(2*r**2))


def set_initial_state(model, samesign=False):
    x0, y0, r0, d = 0.5, 0.5, 0.05, 0.15
    omega = model.state.omega
    u = model.state.u
    b = model.state.b
    msk = model.mesh.msk
    dx = model.mesh.dx
    dy = model.mesh.dy

    x, y = model.mesh.xy()
    u.x[:, :] = 0.2*dx*model.mesh.mskx

    b[:, :] = y * model.mesh.msk

    model.integrator.diag(model.state)


def set_mountain(model):
    msk = model.mesh.msk
    x, y = model.mesh.xy()

    ymountain = gaussian(x, y, 0.5, 0, 0.1)
    msk[y < ymountain] = 0
    model.mesh.finalize()


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    plt.ion()

    param = f2d.Param()

    param.model = "boussinesq"
    param.Lx = 4
    param.ny = 50*2
    param.nx = param.ny*param.Lx
    param.tend = 100
    param.dtmax = 0.1
    param.maxite = 5_000
    param.xperiodic = True

    param.nplot = 20
    param.animation = True
    param.plotvar = "Uy"

    param.nhis = 20
    param.var_to_store = ["omega", "p", "U", "b"]

    model = f2d.Model(param)
    set_mountain(model)
    set_initial_state(model)
    model.run()
