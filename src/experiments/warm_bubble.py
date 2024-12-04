import numpy as np
import fluids2d as f2d


def gaussian(x, y, x0, y0, r): return np.exp(-((x-x0)**2+(y-y0)**2)/(2*r**2))


def forcing(param, mesh, s, ds):
    Q = param.Q
    ds.b[0] += Q
    ds.b[1:-1] -= Q/(mesh.ny-1)


def set_initial_state(model):
    mesh = model.mesh
    x, y = mesh.xy()
    b = model.state.b
    # b[:, :] = y+1e-2*np.random.normal(size=mesh.shape)
    b[:, :] = y + 0.1*gaussian(x, y, 1, 0.25, 0.02)
    b *= mesh.msk


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    plt.ion()

    param = f2d.Param()

    param.model = "boussinesq"
    param.nx = 200
    param.Lx = 2.
    param.ny = 100
    param.tend = 500
    param.maxite = 5_000
    param.cfl = 0.9
    param.dtmax = 1e-1
    param.xperiodic = False

    param.nplot = 10
    param.animation = True
    param.plotvar = "Uy"
    param.clims = [-0.02, 0.02]

    param.nhis = 10
    param.var_to_store = ["b", "omega", "p"]
    param.add_parameter("Q")
    param.Q = 0.05

    model = f2d.Model(param)
    # model.add_forcing(forcing)
    set_initial_state(model)
    model.run()
