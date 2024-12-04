import numpy as np
import fluids2d as f2d
from fluids2d.equations import fill


def step(x, x0, dx): return np.tanh((x-x0)/(3*dx))


def set_initial_buoyancy(model):
    x, y = model.mesh.xy("c")
    b = model.state.b
    Lx = model.param.Lx
    b[:, :] = (-step(x, Lx/4, model.mesh.dx)
               + step(x, Lx/2, model.mesh.dx))
    b *= model.mesh.msk
    fill(model.mesh, b)
    model.integrator.diag(model.state)


def set_model(model="boussinesq"):
    param = f2d.Param()

    param.model = model
    param.nx = 200
    param.Lx = 5.
    param.ny = 40
    param.tend = 50
    param.maxite = 5_000
    param.cfl = 0.9
    param.dtmax = 1.
    param.integrator = "rk3"
    param.maxorder = 6
    param.xperiodic = True

    param.nhis = 10
    param.var_to_store = ["b", "omega", "U"]

    param.compflux = "weno"
    param.vortexforce = "weno"
    param.innerproduct = "weno"

    param.nplot = 10
    param.animation = True
    param.plotvar = "omega"
    # param.clims = [-1, 1]
    param.clims = np.asarray([-0.0012, 0.0012])

    model = f2d.Model(param)
    set_initial_buoyancy(model)
    return model


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    plt.ion()

    twinexp = True

    model = set_model()

    if twinexp:
        model2 = set_model(model="hydrostatic")
        param = model2.param
        param.maxorder = 4

        param.compflux = "upwind"
        param.vortexforce = "upwind"
        param.innerproduct = "upwind"

        f2d.tools.run_twin_experiments(model2, model)

    else:
        model.run()
