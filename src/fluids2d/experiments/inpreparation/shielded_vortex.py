import numpy as np
import fluids2d as f2d


def set_initial_state(model, amp_noise=1e-4):
    x0, y0, r0 = 0.5, 0.5, 0.06
    omega = model.state.omega
    dx = model.mesh.dx
    dy = model.mesh.dy
    area = dx*dy

    def set_omega(omega, msk):

        r1 = 1.4*r0

        z = (x-0.5)+1j*(y-0.5)
        d = np.abs(z)
        inner = 1*(d <= r0)
        outer = 1*(d <= r1)
        ratio = inner.sum()/outer.sum()

        omega[:, :] = (outer*ratio - inner)

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
    param.ny = 50*4
    param.nx = param.ny
    param.tend = 100
    param.maxite = 5_000
    param.cfl = 0.9

    param.nplot = 20
    param.animation = True
    param.plotvar = "omega"

    param.nhis = 20
    param.var_to_store = ["omega", "U"]

    model = f2d.Model(param)
    set_initial_state(model, amp_noise=1e-9)
    model.run()
