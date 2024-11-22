import numpy as np
from PIL import Image
import fluids2d as f2d
from fluids2d.integrators import copyto


def gaussian(x, y, x0, y0, r): return np.exp(-((x-x0)**2+(y-y0)**2)/(2*r**2))


def set_initial_velocity(model, flow="dipole"):
    mesh = model.mesh
    x, y = mesh.xy("v")
    x0, y0, r0, d = 0.5, 0.5, 0.1, 0.1
    U = model.state.U
    if flow == "dipole":
        omega = gaussian(x, y, x0-d, y0, r0)-gaussian(x, y, x0+d, y0, r0)
    else:
        omega = (4*np.pi/100)*np.ones(mesh.shape)

    omega *= mesh.mskv
    f2d.tools.set_uv_from_omega(model, omega, U)


def set_initial_vector(model):
    x, y = model.mesh.xy("x")
    vx = model.state.v.x
    vx[:, :] = gaussian(x, y, 0.7, 0.5, 0.05)
    vx *= model.mesh.mskx


def fliptime(model):
    s = model.state
    if model.integrator == "LFRA":
        sb, sa, ds = model.integrator.scratch
        copyto(s, sa)
        copyto(sb, s)
        copyto(sa, s)
        model.integrator.diag(s)

    s.U.x[:] *= -1
    s.U.y[:] *= -1
    model.time.t = 0


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    # plt.ion()

    param = f2d.Param()

    param.model = "vectoradv"
    param.nx = 50
    param.ny = 50
    param.tend = 100
    param.maxite = 5_000
    param.cfl = 0.9
    param.nplot = 5
    param.animation = True
    param.plotvar = "v"
    param.clims = [-1, 1]
    # param.integrator = "LFRA"
    # param.maxorder = 2
    # param.compflux = "centered"
    # param.vortexforce = "centered"
    # param.innerproduct = "centered"

    model = f2d.Model(param)

    if True:
        x, y = model.mesh.xy()
        r2 = (x-0.5)**2+(y-0.5)**2
        model.mesh.msk[r2 > 0.5**2] = 0
        # model.mesh.msk[:param.ny//3, :param.nx//3] = 0
        model.mesh.finalize()

    set_initial_velocity(model, flow="body")

    set_initial_vector(model)

    model.run()
