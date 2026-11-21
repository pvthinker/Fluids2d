import numpy as np
import fluids2d as f2d


def gaussian(x, y, x0, y0, r): return np.exp(-((x-x0)**2+(y-y0)**2)/(2*r**2))


def set_initial_state(model, amp=0.1, length=0.5):
    x0, y0, r0, d = 0.5, 0.5, 0.05, 0.15
    omega = model.state.omega
    u = model.state.u
    b = model.state.b
    msk = model.mesh.msk
    dx = model.mesh.dx
    dy = model.mesh.dy

    def density_profile(y):
        return np.tanh((y-y0)/(3*dy))

    def perturbation(x):
        if x > np.pi:
            return 0.0
        else:
            return (np.cos(x)+1)/2

    f = np.vectorize(perturbation)

    x, y = model.mesh.xy()

    perturb = f(np.pi*x/length)

    b[:, :] = density_profile(y+amp*perturb)
    b *= model.mesh.msk

    model.integrator.diag(model.state)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    plt.ion()

    param = f2d.Param()

    param.model = "boussinesq"
    param.Lx = 2
    param.ny = 50*2
    param.nx = param.ny*param.Lx
    param.tend = 10
    param.dtmax = 0.1
    param.maxite = 5_000

    param.nplot = 20
    param.animation = True
    param.plotvar = "b"

    param.nhis = 20
    param.var_to_store = ["omega", "p", "U", "b"]

    model = f2d.Model(param)

    set_initial_state(model, amp=1, length=0.5)
    model.run()
