import numpy as np
from PIL import Image
import fluids2d as f2d
from fluids2d.integrators import copyto


def gaussian(x, y, x0, y0, r): return np.exp(-((x-x0)**2+(y-y0)**2)/(2*r**2))


def set_initial_velocity(model, flow="dipole"):
    mesh = model.mesh
    x, y = mesh.xy("v")
    x0, y0, r0, d = 0.5, 0.5, 0.05, 0.05
    U = model.state.U
    if flow == "dipole":
        omega = gaussian(x, y, x0-d, y0, r0)-gaussian(x, y, x0+d, y0, r0)
    elif flow == "bodyrotation":
        omega = (4*np.pi/100)*np.ones(mesh.shape)
    elif flow == "vortex":
        omega = gaussian(x, y, x0, y0, r0)
    else:
        raise ValueError

    omega *= mesh.mskv
    f2d.tools.set_uv_from_omega(model, omega, U)


def set_tracer_from_image(model, imagefile="../data/cow.png"):
    ny, nx = model.mesh.ny, model.mesh.nx
    im = Image.open(imagefile).resize((ny, nx))
    tracer = np.zeros((ny, nx))
    red = im.getchannel("R")
    green = im.getchannel("G")
    blue = im.getchannel("B")
    tracer += red
    tracer += green
    tracer += blue
    tracer = np.flipud(tracer/3)
    tracer = (tracer/256)
    model.state.q[:-1, :-1] = tracer
    model.state.q[:, :] *= model.mesh.msk


def set_initial_tracer(model):
    x, y = model.mesh.xy("c")
    q = model.state.q
    q[:, :] = (np.round(x*8) % 2 + np.round(y*8) % 2)/2
    q *= model.mesh.msk


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

    param = f2d.Param()

    param.model = "advection"
    param.nx = 100
    param.ny = 100
    param.tend = 100
    param.maxite = 5_000
    param.cfl = 0.9
    param.nplot = 5
    param.animation = True
    param.plotvar = "q"
    param.integrator = "rk3"

    if param.integrator == "LFRA":
        param.RAgamma = 0.
        param.cfl = 0.5
        param.compflux = "centered"

    param.cmap = "Greys_r"

    param.nhis = 10
    param.var_to_store = ["q"]

    model = f2d.Model(param)

    x, y = model.mesh.xy()

    circular_domain = False
    add_obstacle = False

    if circular_domain:
        r2 = (x-0.5)**2+(y-0.5)**2
        model.mesh.msk[r2 > 0.5**2] = 0

    if add_obstacle:
        model.mesh.msk[:param.ny//3, :param.nx//3] = 0

    model.mesh.finalize()

    set_initial_velocity(model, flow="vortex")

    set_initial_tracer(model)

    # set_tracer_from_image(model)

    model.run()
