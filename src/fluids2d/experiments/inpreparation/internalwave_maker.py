import numpy as np
import fluids2d as f2d


def gaussian(x, y, x0, y0, r): return np.exp(-((x-x0)**2+(y-y0)**2)/(2*r**2))


def set_forcing(model, amp=0.1, period=10):
    mesh = model.mesh
    x, y = mesh.xy()
    omega = 2*np.pi/period
    perturb = gaussian(x, y, 0.5, 0.5, 0.03)*mesh.dx**2/omega*mesh.msky
    perturb *= amp
    mesh .time = model.time

    def forcing(param, mesh, s, ds):
        ds.u.y[:, :] += perturb*np.sin(mesh.time.t*omega)

    model.add_forcing(forcing)


def set_initial_state(model):
    x0, y0, r0, d = 0.5, 0.5, 0.05, 0.15
    b = model.state.b
    msk = model.mesh.msk

    x, y = model.mesh.xy()

    b[:, :] = y
    b *= model.mesh.msk

    model.integrator.diag(model.state)


def set_bottom_topography(model):
    msk = model.mesh.msk
    x, y = model.mesh.xy()

    ytop = 0.5
    slope = 1
    x0 = model.param.Lx/2

    f = np.vectorize(lambda x: np.maximum(0, ytop-slope*np.abs(x-x0)))
    ymountain = f(x)

    msk[y < ymountain] = 0
    model.mesh.finalize()


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    plt.ion()

    param = f2d.Param()

    param.model = "boussinesq"
    param.Lx = 2
    param.ny = 50*2
    param.nx = param.ny*param.Lx
    param.tend = 200
    param.dtmax = 0.1
    param.maxite = 5_000
    param.xperiodic = True

    param.nplot = 20
    param.animation = True
    param.plotvar = "b"

    param.nhis = 20
    param.var_to_store = ["omega", "U", "b"]

    model = f2d.Model(param)

    set_forcing(model, amp=2, period=10)
    set_initial_state(model)
    model.run()
