import numpy as np
import fluids2d as f2d


def gaussian(x, y, x0, y0, r): return np.exp(-((x-x0)**2+(y-y0)**2)/(2*r**2))


def set_initial_dipole(model, samesign=False):
    x0, y0, r0, d = 0.5, 0.5, 0.05, 0.15
    omega = model.state.omega

    def dipole(x, y):
        return gaussian(x, y, x0+d, y0, r0)+gaussian(x, y, x0-d, y0, r0)

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

    model.integrator.diag(model.state)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    plt.ion()

    param = f2d.Param()

    param.model = "euler"
    param.nx = 50*4
    param.ny = 50*4
    param.tend = 100
    param.maxite = 5_000
    param.cfl = 0.9
    param.integrator = "rk3"
    param.xperiodic = False

    param.nplot = 20
    param.animation = True
    param.plotvar = "omega"

    param.nhis = 20
    param.var_to_store = ["omega", "p"]

    model = f2d.Model(param)
    set_initial_dipole(model)
    model.run()
