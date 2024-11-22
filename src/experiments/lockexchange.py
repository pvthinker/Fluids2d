import numpy as np
import fluids2d as f2d


def set_initial_buoyancy(model):
    x, y = model.mesh.xy("c")
    b = model.state.b
    Lx = model.param.Lx
    b[:, :] = -np.tanh((x-Lx/2)/(3*model.mesh.dx))
    b *= model.mesh.msk
    model.integrator.diag(model.state)


def set_model(model="boussinesq"):
    param = f2d.Param()

    param.model = model
    param.nx = 200
    param.Lx = 5.
    param.ny = 40
    param.tend = 20
    param.maxite = 2_000
    param.cfl = 0.9
    param.dtmax = 1.
    param.integrator = "rk3"
    param.maxorder = 6

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

    twinexp = False

    model = set_model()

    if twinexp:
        model2 = set_model()
        param = model2.param
        param.maxorder = 4

        # param.compflux = "weno"
        param.vortexforce = "centered"
        param.innerproduct = "classic"

        f2d.tools.run_twin_experiments(model2, model)

    else:
        model.run()
