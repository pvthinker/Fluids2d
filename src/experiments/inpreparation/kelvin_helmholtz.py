import numpy as np
import fluids2d as f2d


def gaussian(x, y, x0, y0, r): return np.exp(-((x-x0)**2+(y-y0)**2)/(2*r**2))


def set_initial_state(model, speed=1):
    x0, y0, r0 = 0.5, 0.5, 0.05
    omega = model.state.omega
    u = model.state.u
    b = model.state.b
    msk = model.mesh.msk
    dx = model.mesh.dx
    dy = model.mesh.dy

    def jetprofile(x, y):
        # return gaussian(x0, y, x0, y0, r0)*(y-y0)/r0**2
        return np.tanh((y-y0)/r0)

    assert model.param.model == "boussinesq"

    x, y = model.mesh.xy("v")
    pertub = np.sin(x*np.pi*2)
    yy = y+1e-3*pertub
    # omega[:, :] = speed*jetprofile(x, yy)
    # noise = np.random.normal(size=omega.shape)
    # omega *= (1+0.05*noise)
    # omega *= model.mesh.mskv*model.mesh.area
    # omega *= 0.2

    xc, yc = model.mesh.xy()
    b[:, :] = yc

    u.x[:, :] = speed*jetprofile(x, yy)
    u.x[:, :] *= model.mesh.msk*dx
    # f2d.tools.set_uv_from_omega(model, omega, u)

    model.integrator.diag(model.state)


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
    param.cfl = 0.9
    param.f0 = 0
    param.xperiodic = True

    param.nplot = 20
    param.animation = True
    param.plotvar = "omega"

    param.nhis = 20
    param.var_to_store = ["omega", "p", "U", "b"]

    model = f2d.Model(param)
    set_initial_state(model, speed=0.2)
    model.run()
