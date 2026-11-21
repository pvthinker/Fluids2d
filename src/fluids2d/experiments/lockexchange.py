import numpy as np
import fluids2d as f2d
from fluids2d.equations import fill


def step(x, x0, dx): return np.tanh((x-x0)/(3*dx))


def set_initial_buoyancy(model):
    x, y = model.mesh.xy("c")
    b = model.state.b
    Lx = model.param.Lx
    b[:, :] = step(x, Lx/2, model.mesh.dx) + 1e-8*np.random.normal(size=b.shape)
    b *= model.mesh.msk
    fill(model.mesh, b)
    model.integrator.diag(model.state)


def set_model(model="boussinesq"):
    param = f2d.Param()

    param.model = model
    param.Lx = 5.
    param.ny = 40
    param.nx = int(param.ny*param.Lx)
    param.tend = 8
    param.maxite = 5_000
    param.cfl = 0.9
    param.dtmax = 0.02
    param.integrator = "rk3"
    param.maxorder = 6
    param.xperiodic = False

    param.nhis = 20
    param.var_to_store = ["b", "omega", "U"]

    param.compflux = "weno"
    param.vortexforce = "weno"
    param.innerproduct = "weno"

    param.nplot = 10
    param.animation = True
    param.plotvar = "b"
    param.clims = np.asarray([-1, 1])

    model = f2d.Model(param)
    set_initial_buoyancy(model)
    return model


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    plt.ion()

    twinexp = False

    model = set_model(model="boussinesq")

    if twinexp:
        model2 = set_model(model="hydrostatic")
        #
        # tweak below the parameters of the second experiment
        #
        # param = model2.param
        # param.maxorder = 6

        # param.compflux = "weno"
        # param.vortexforce = "weno"
        # param.innerproduct = "weno"

        f2d.tools.run_twin_experiments(model2, model, hstack=True)

    else:
        model.run()
