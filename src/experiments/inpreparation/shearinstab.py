import numpy as np
import fluids2d as f2d


def gaussian(x, y, x0, y0, r): return np.exp(-((x-x0)**2+(y-y0)**2)/(2*r**2))


def set_initial_state(model, amp_noise=1e-4):
    x0, y0, r0 = 0.5, 0.5, 0.04
    omega = model.state.omega
    dx = model.mesh.dx
    dy = model.mesh.dy
    area = dx*dy

    def jetprofile(x, y):
        return gaussian(x0, y, x0, y0, r0)*(y-y0)/r0**2

    def set_omega(omega, msk):
        omega[:, :] = jetprofile(x, y)
        noise = np.random.normal(size=omega.shape)
        omega *= (1+amp_noise*noise)

        omega *= msk*model.mesh.area

    if model.param.model == "euler":
        x, y = model.mesh.xy("v")
        set_omega(omega, model.mesh.mskv)
        u = model.state.u
        f2d.tools.set_uv_from_omega(model, omega, u)

    elif model.param.model == "eulerpsi":
        x, y = model.mesh.xy("c")
        set_omega(omega, model.mesh.msk)

    model.integrator.diag(model.state)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    plt.ion()

    param = f2d.Param()

    param.model = "euler"
    param.Lx = 4
    param.ny = 50*2
    param.nx = param.ny*param.Lx
    param.tend = 100
    param.maxite = 5_000
    param.cfl = 0.9
    param.xperiodic = True

    param.nplot = 20
    param.animation = True
    param.plotvar = "omega"

    param.nhis = 20
    param.var_to_store = ["omega", "U"]

    model = f2d.Model(param)
    set_initial_state(model)
    model.run()
