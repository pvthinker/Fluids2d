import numpy as np
import fluids2d as f2d


def set_forcing(model, Q=5e-2, setup="RB", warmingfrac=0.5):
    model.mesh.time = model.time
    nh = model.param.halowidth
    if setup == "RB":
        def forcing(param, mesh, s, ds):
            ds.b[nh, nh:-nh] += Q
            ds.b[nh+1:-nh, nh:-nh] -= Q/(mesh.ny-1)

    elif setup == "HorizConv":
        imid = int(model.param.nx*warmingfrac)
        jtop = -1
        dy = model.mesh.dy
        Qcool = Q*warmingfrac/(1-warmingfrac)/dy
        Qwarm = Q/dy

        def forcing(param, mesh, s, ds):
            t = mesh.time.t
            coef = np.tanh(t/50)
            ds.b[jtop-nh:-nh, nh:imid] += Qwarm*coef
            ds.b[jtop-nh:-nh, imid:-nh] -= Qcool*coef

    model.add_forcing(forcing)


def set_initial_state(model):
    mesh = model.mesh
    x, y = mesh.xy()
    b = model.state.b
    b[:, :] = 10*y+1e-2*np.random.normal(size=mesh.shape)
    b *= mesh.msk


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    plt.ion()

    param = f2d.Param()

    param.model = "boussinesq"
    param.Lx = 4
    param.ny = 25*2
    param.nx = param.ny*param.Lx
    param.tend = 1_000
    param.maxite = 50_000
    param.cfl = 0.9
    param.dtmax = 1e-1
    param.xperiodic = False
    param.noslip = ["bottom"]

    param.nplot = 10
    param.animation = True
    param.plotvar = "b"
    # param.clims = [-10, 10.]

    param.nhis = 100
    param.var_to_store = ["b", "U"]

    model = f2d.Model(param)

    # set_forcing(model, setup="HorizConv", Q=0.1, warmingfrac=0.9)
    set_forcing(model, Q=1)

    set_initial_state(model)
    model.run()
