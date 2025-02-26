import numpy as np
import fluids2d as f2d


def gaussian(x, y, x0, y0, r): return np.exp(-((x-x0)**2+(y-y0)**2)/(2*r**2))


def set_initial_dipole(model, samesign=False):
    Lx = model.param.Lx
    x0, y0, r0, d = Lx/2, 0.5, 0.05, 0.05
    omega = model.state.omega

    def dipole(x, y):
        return gaussian(x, y, x0+d, y0, r0)-gaussian(x, y, x0-d, y0, r0)

    if model.param.model == "euler":
        x, y = model.mesh.xy("v")
        u = model.state.u
        omega[:, :] = dipole(x, y)
        omega *= model.mesh.mskv*model.mesh.area
        f2d.tools.set_uv_from_omega(model, omega, u)

    elif model.param.model == "eulerpsi":
        x, y = model.mesh.xy("c")
        omega[:, :] = dipole(x, y)
        omega *= model.mesh.msk*model.mesh.area

    elif model.param.model == "rsw":
        x, y = model.mesh.xy("v")
        u = model.state.u

        h = model.state.h
        dx = model.mesh.dx
        dy = model.mesh.dy
        area = dx*dy
        h[:, :] = 1*model.mesh.msk*area

        omega[:, :] = dipole(x, y)
        omega *= model.mesh.mskv*model.mesh.area
        f2d.tools.set_uv_from_omega(model, omega, u)

    model.integrator.diag(model.state)


def set_mask(model):
    p = model.param
    x, y = model.mesh.xy()
    msk = model.mesh.msk
    msk[y < 0.2-0.5*np.abs(x-p.Lx/2)] = 0

    model.mesh.finalize()


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    plt.ion()

    param = f2d.Param()

    param.model = "euler"
    param.Lx = 2.0
    param.ny = 100
    param.nx = param.ny*int(param.Lx/param.Ly)
    param.tend = 200
    param.dt = 0.1*(200/param.ny)
    param.f0 = 0.
    param.maxite = 5_000
    param.cfl = 0.9
    param.xperiodic = False
    param.noslip = False

    param.nplot = 5
    param.animation = True
    param.generate_mp4 = True
    param.plotvar = "omega"

    param.nhis = 25
    param.var_to_store = ["omega", "p", "U"]

    model = f2d.Model(param)

    # set_mask(model)

    set_initial_dipole(model)

    model.run()
