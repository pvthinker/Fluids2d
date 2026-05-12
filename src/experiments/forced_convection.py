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
        nx = model.param.nx
        imid = int(nx*warmingfrac)
        jtop = -1
        dy = model.mesh.dy
        Qcool = (Q/dy)*imid/(nx-imid)
        Qwarm = Q/dy
        Qforc = model.state.b*0
        Qforc[jtop-nh:-nh, nh:nh+imid] = Qwarm
        Qforc[jtop-nh:-nh, nh+imid:-nh] = -Qcool

        def forcing(param, mesh, s, ds):
            t = mesh.time.t
            coef = 1#np.tanh(t*Q)
            ds.b[jtop-nh:-nh, nh:nh+imid] += Qwarm*coef
            ds.b[jtop-nh:-nh, nh+imid:-nh] -= Qcool*coef

    model.add_forcing(forcing)
    model.Qforc = Qforc


def set_initial_state(model):
    mesh = model.mesh
    x, y = mesh.xy()
    b = model.state.b
    b[:, :] = 3*y+1e-2*np.random.normal(size=mesh.shape)
    b *= mesh.msk


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    plt.ion()

    Q = 0.8

    param = f2d.Param()

    param.model = "boussinesq"
    param.Lx = 4
    param.ny = 25*2
    param.nx = param.ny*param.Lx
    param.tend = 4/Q
    param.maxite = 500_000
    param.cfl = 0.9
    param.dtmax = 1e-2/Q
    param.xperiodic = False
    param.noslip = ["bottom"]
    param.integrator = "rk3"

    param.nplot = 200
    param.animation = True
    param.plotvar = "b"
    # param.clims = [-10, 10.]

    param.nhis = 10
    param.var_to_store = ["b", "U"]

    model = f2d.Model(param)

    np.random.seed(42)
    set_forcing(model, setup="HorizConv", Q=Q, warmingfrac=0.95)

    set_initial_state(model)
    model.state.b[:,:] *= (10*Q)
    #model.run()
