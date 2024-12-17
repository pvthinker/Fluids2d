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


def get_diag_ke(model):
    s = model.state
    ke0 = (np.sum(s.U.x**2)+np.sum(s.U.y**2))

    def diag_ke(param, mesh, s, time):
        ke = (np.sum(s.U.x**2)+np.sum(s.U.y**2))-ke0
        ke /= ke0
        print(f"  {ke:.4}", end="")

    return diag_ke


def fliptime(model):
    s = model.state
    if model.param.model == "eulerpsi":
        s.omega[:] *= -1
    else:
        s.u.x[:] *= -1
        s.u.y[:] *= -1
    model.integrator.diag(s)
    model.time.t = 0
    model.time.ite = 0


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    plt.ion()

    param = f2d.Param()

    param.model = "euler"
    param.nx = 50*2
    param.ny = 50*2
    param.tend = 100
    param.maxite = 100  # 5_000
    # param.cfl = 0.4
    param.dtmax = 0.125
    param.integrator = "rk3"
    param.xperiodic = False
    param.maxorder = 6
    param.compflux = "centered"
    param.vortexforce = "centered"
    param.innerproduct = "centered"

    param.nplot = 20
    param.animation = True
    param.plotvar = "omega"

    param.nhis = 20
    param.var_to_store = ["omega"]

    model = f2d.Model(param)
    set_initial_dipole(model)
    model.callbacks += [get_diag_ke(model)]

    omega = model.state.omega
    omega0 = omega.copy()

    model.run()
    fliptime(model)
    model.run()
    fliptime(model)
    # nt = 100
    # model.step(nt)
    # fliptime(model)
    # model.step(nt)
    # fliptime(model)

    plt.clf()
    domega = (omega-omega0)/model.mesh.area
    plt.imshow(domega)
    plt.colorbar()
