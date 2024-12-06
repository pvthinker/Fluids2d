import numpy as np
from .states import Vector
from . import weno


def addvortexforce(param, mesh, U, omega, du):
    method = param.vortexforce
    if isinstance(du, Vector):
        weno.vortexforce(du.x, U.y, omega, mesh.ov.y,
                         mesh.yshift, mesh.xshift, +1, method)

        weno.vortexforce(du.y, U.x, omega, mesh.ov.x,
                         mesh.xshift, mesh.yshift, -1, method)

    else:
        # hydrostatic case
        weno.vortexforce(du, U.y, omega, mesh.ov.y,
                         mesh.yshift, mesh.xshift, +1, method)


def divflux(param, mesh, flx, q, U, dq):
    method = param.compflux
    weno.compflux(flx.x, U.x, q, mesh.oc.x, mesh.xshift,
                  method, param.nthreads)

    weno.compflux(flx.y, U.y, q, mesh.oc.y, mesh.yshift,
                  method, param.nthreads)

    div(mesh, flx, dq)


def addcoriolis(param, mesh, U, du):
    f = param.f0*mesh.area*0.25

    du.x[:-1, 1:-1] += f*(U.y[:-1, :-2]+U.y[1:, :-2] +
                          U.y[:-1, 1:-1]+U.y[1:, 1:-1])

    du.y[1:-1, :-1] -= f*(U.x[:-2, :-1]+U.x[:-2, 1:] +
                          U.x[1:-1, :-1]+U.x[1:-1, 1:])


def compute_pv(param, mesh, omega, h, pv):
    f = param.f0*mesh.area
    centerstovertices(mesh, h, pv)
    pv[pv > 0] = (f+omega[pv > 0])/pv[pv > 0]


def addgrad(mesh, phi, du):
    if isinstance(du, Vector):

        du.x[:, 1:] -= np.diff(phi, axis=1)*mesh.mskx[:, 1:]
        du.y[1:, :] -= np.diff(phi, axis=0)*mesh.msky[1:, :]

    else:
        du[:, 1:] -= np.diff(phi, axis=1)*mesh.mskx[:, 1:]


def sharp(mesh, u, U):
    if isinstance(u, Vector):
        U.x[:] = u.x*(1/mesh.dx**2)
        U.y[:] = u.y*(1/mesh.dy**2)
    else:
        U.x[:] = u*(1/mesh.dx**2)


def compute_vorticity(mesh, u, omega):
    if isinstance(u, Vector):

        omega[1:, :] = -np.diff(u.x, axis=0)
        omega[:, 1:] += np.diff(u.y, axis=1)

    else:

        omega[1:, :] = -np.diff(u, axis=0)

    omega *= mesh.mskv


def compute_kinetic_energy(param, mesh, u, U, ke):
    method = param.innerproduct

    ke[:] = 0.

    if isinstance(u, Vector):
        if method == "classic":
            ke[:, :-1] = +u.x[:, 1:]*U.x[:, 1:] + u.x[:, :-1]*U.x[:, :-1]
            ke[:-1, :] += u.y[1:, :]*U.y[1:, :] + u.y[:-1, :]*U.y[:-1, :]
            ke *= mesh.msk*0.25
        else:
            weno.innerproduct(ke, U.x, u.x, mesh.ok.x, mesh.xshift, method)
            weno.innerproduct(ke, U.y, u.y, mesh.ok.y, mesh.yshift, method)
            ke *= mesh.msk*0.5

    else:
        if method == "classic":
            ke[:, :-1] = +u[:, 1:]*U.x[:, 1:] + u[:, :-1]*U.x[:, :-1]
            ke *= mesh.msk*0.25
        else:
            weno.innerproduct(ke, U.x, u, mesh.ok.x, mesh.xshift, method)
            ke *= mesh.msk*0.5


def div(mesh, U, delta):
    delta[:, :-1] = -np.diff(U.x, axis=1)
    delta[:-1, :] -= np.diff(U.y, axis=0)
    delta *= mesh.msk


def compute_pressure(param, mesh, h, p):
    p[:] = (param.g/mesh.area) * h


def pressure_projection(mesh, U, delta, p, u):
    sharp(mesh, u, U)
    div(mesh, U, delta)
    mesh.poisson_centers.solve(-delta*mesh.area, p)
    addgrad(mesh, p, u)
    mesh.fill(u)


def compute_streamfunction(mesh, vomega, psi):
    mesh.poisson_vertices.solve(vomega, psi)


def centerstovertices(mesh, omega, vomega, addto=False):
    if addto:
        vomega[1:, 1:] += 0.25*(omega[:-1, :-1]+omega[1:, :-1] +
                                omega[:-1, 1:]+omega[1:, 1:])
    else:
        vomega[1:, 1:] = 0.25*(omega[:-1, :-1]+omega[1:, :-1] +
                               omega[:-1, 1:]+omega[1:, 1:])
    vomega *= mesh.mskv


def verticestocenters(mesh, vh, h):
    m = mesh.mskv
    coef = m[:-1, :-1]+m[1:, :-1] + m[:-1, 1:]+m[1:, 1:]
    h[:-1, :-1] = (1/coef)*(vh[:-1, :-1]+vh[1:, :-1] +
                            vh[:-1, 1:]+vh[1:, 1:])
    h *= mesh.msk


def perpgrad(mesh, psi, u, contravariant=False):
    u.x[:-1, :] = -np.diff(psi, axis=0)*mesh.mskx[:-1, :]
    u.y[:, :-1] = np.diff(psi, axis=1)*mesh.msky[:, :-1]
    if contravariant:
        u.x[:] *= (1/mesh.dy**2)
        u.y[:] *= (1/mesh.dx**2)


def addbuoyancy(mesh, b, du):
    du.y[1:, :] += (0.5*mesh.area)*(b[1:, :]+b[:-1, :])*mesh.msky[1:, :]


def compute_vertical_velocity(mesh, U):
    U.y[1:, :-1] = -np.cumsum(np.diff(U.x[:-1, :], axis=1), axis=0)


def compute_hydrostatic_pressure(mesh, b, p):
    p[:, :] = 0.5*b
    p[-1::-1, :] -= np.cumsum(b[-1::-1, :], axis=0)
    p *= (mesh.msk*mesh.dx**2)


def apply_pressure_surface_correction(mesh, U, uh):
    # U.y[-1] should not be masked
    # otherwise the poisson solve will return ps=0
    ps = mesh.poisson1d.solve(-U.y[-1])
    mesh.fill(ps)
    uh[:, 1:] -= np.diff(ps)
    uh *= mesh.mskx
    return ps


def qg_projection(mesh, u, h, pv, psi, anomaly=False):
    f0 = mesh.param.f0
    H = mesh.param.H
    compute_vorticity(mesh, u, pv)
    add_stretching(mesh, pv, h, anomaly)
    mesh.qg_helmholtz.solve(pv, psi)
    thickness_from_psi(mesh, psi, h, anomaly)
    perpgrad(mesh, psi, u)


def qg_inversion(mesh, pv, work, psi):
    centerstovertices(mesh, pv, work)
    mesh.qg_helmholtz.solve(work, psi)


def add_stretching(mesh, pv, h, anomaly):
    f0 = mesh.param.f0
    H = mesh.param.H
    h0 = H*mesh.area
    if anomaly:
        centerstovertices(mesh, h*(-f0/H), pv, addto=True)
    else:
        centerstovertices(mesh, (h-h0)*(-f0/H), pv, addto=True)


def thickness_from_psi(mesh, psi, h, anomaly):
    f0 = mesh.param.f0
    H = mesh.param.H
    g = mesh.param.g
    verticestocenters(mesh, psi * (f0*mesh.area/g), h)
    if not anomaly:
        h += (H*mesh.area)
    h *= mesh.msk
