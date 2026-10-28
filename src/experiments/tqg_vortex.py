import numpy as np
import fluids2d as f2d
from fluids2d.operators import compute_vorticity, verticestocenters, perpgrad

"""
The TQG vortex instability from Carton et al. 2025 (figure 4)

https://doi.org/10.3390/fluids10110280

"""

def gaussian(x, y, r): return np.exp(-(x**2+y**2)/(2*r**2))

def compute_Rd(param):
    return param.f0**2/(param.g*param.H)

def compute_qgpv_from_psi(mesh, pv, psi, U):
    Rd = compute_Rd(mesh.param)
    work = pv[:,:]*0
    perpgrad(mesh, psi, U)
    compute_vorticity(mesh, U, work) # work is on vertices
    work -= psi*(1/Rd**2)
    verticestocenters(mesh, work, pv) # pv is on centers


def set_initial_dipole(model, samesign=False):
    Lx = model.param.Lx
    Ly = model.param.Ly
    x0, y0, r0 = Lx/2, Ly/2, Lx/10

    b0_over_psi0 = 0.3

    xc, yc = model.mesh.xy("c")
    xv, yv = model.mesh.xy("v")

    psi = model.state.psi
    pv = model.state.pv
    b = model.state.b
    U = model.state.U

    eps = 0.05
    initial_profile = lambda x,y: gaussian((1+eps)*(x-x0),(1-eps)*(y-y0),r0)

    psi[:,:] = initial_profile(xv,yv) # at vertices
    psi[:,:] *= model.mesh.area/model.mesh.dx
    # notice the 1/dx term in psi to make pv and b comparable in magnitude
    compute_qgpv_from_psi(model.mesh, pv, psi, U)

    b[:,:] = b0_over_psi0 * initial_profile(xc,yc)*model.mesh.area # at centers


    model.integrator.diag(model.state)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    plt.ion()

    param = f2d.Param()

    param.model = "tqg"
    param.Lx = 1.0
    param.ny = 200
    param.nx = param.ny*int(param.Lx/param.Ly)
    param.tend = 200
    param.dt = 0.1*(200/param.ny)
    param.f0 = 10.
    param.g = 1
    param.H = 1
    param.maxite = 2000
    param.cfl = 0.9
    param.xperiodic = False
    param.noslip = False

    param.nplot = 5
    param.animation = True
    param.generate_mp4 = False
    param.plotvar = "b"

    param.nhis = 25
    param.var_to_store = ["pv", "b", "psi", "U"]

    model = f2d.Model(param)

    set_initial_dipole(model)

    model.run()

