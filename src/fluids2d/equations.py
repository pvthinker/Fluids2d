from .operators import *


def fill(mesh, *variables):
    for v in variables:
        mesh.fill(v)


def get_euler(param, mesh):

    def rhs(s, ds):
        """ RHS for Euler model in momentum-pressure"""
        addvortexforce(param, mesh, s.U, s.omega, ds.u)
        addgrad(mesh, s.ke, ds.u)
        fill(mesh, ds.u)

    def diag(s):
        pressure_projection(mesh, s.U, s.div, s.p, s.u)
        sharp(mesh, s.u, s.U)
        compute_vorticity(mesh, s.u, s.omega)
        compute_kinetic_energy(param, mesh, s.u, s.U, s.ke)
        fill(mesh, s.omega, s.ke)

    return (rhs, diag)


def get_boussinesq(param, mesh):

    def rhs(s, ds):
        """ RHS for Boussinesq model in momentum-pressure"""
        addvortexforce(param, mesh, s.U, s.omega, ds.u)
        addgrad(mesh, s.ke, ds.u)
        addbuoyancy(mesh, s.b, ds.u)
        divflux(param, mesh, s.flx, s.b, s.U, ds.b)
        fill(mesh, ds.u, ds.b)

    def diag(s):
        pressure_projection(mesh, s.U, s.div, s.p, s.u)
        sharp(mesh, s.u, s.U)
        compute_vorticity(mesh, s.u, s.omega)
        compute_kinetic_energy(param, mesh, s.u, s.U, s.ke)
        fill(mesh, s.omega, s.ke)

    return (rhs, diag)


def get_hydrostatic(param, mesh):

    def rhs(s, ds):
        """ RHS for Hydrostatic model"""
        addvortexforce(param, mesh, s.U, s.omega, ds.uh)
        addgrad(mesh, s.ke, ds.uh)
        addgrad(mesh, s.p, ds.uh)
        divflux(param, mesh, s.flx, s.b, s.U, ds.b)
        fill(mesh, ds.uh, ds.b)

    def diag(s):
        sharp(mesh, s.uh, s.U)
        compute_vertical_velocity(mesh, s.U)
        apply_pressure_surface_correction(mesh, s.U, s.uh)
        sharp(mesh, s.uh, s.U)
        compute_vertical_velocity(mesh, s.U)
        compute_hydrostatic_pressure(mesh, s.b, s.p)
        compute_vorticity(mesh, s.uh, s.omega)
        compute_kinetic_energy(param, mesh, s.uh, s.U, s.ke)
        fill(mesh, s.omega, s.ke)

    return (rhs, diag)


def get_eulerpsi(param, mesh):

    def rhs(s, ds):
        """ RHS for Euler model in vorticity-stream function"""
        divflux(param, mesh, s.flx, s.omega, s.U, ds.omega)
        fill(mesh, ds.omega)

    def diag(s):
        centerstovertices(mesh, s.omega, s.vomega)
        compute_streamfunction(mesh, s.vomega, s.psi)
        perpgrad(mesh, s.psi, s.U, contravariant=True)
        fill(mesh, s.U)

    return (rhs, diag)


def get_qg(param, mesh):

    def rhs(s, ds):
        """RHS for QG model using PV transport and streamfunction

        PV is defined on cell centers as a Finite Volume
        """
        divflux(param, mesh, s.flx, s.pv, s.U, ds.pv)
        fill(mesh, ds.pv)

    def diag(s):
        qg_inversion(mesh, s.pv, s.work, s.psi)
        perpgrad(mesh, s.psi, s.U, contravariant=True)

    return (rhs, diag)


def get_qgrsw(param, mesh):

    def rhs(s, ds):
        """RHS for QG model using the RSW projection method

        see Thiry et al. 2024

        The RSW tendency is projected onto the QG manifold.
        The gradient force projects to zero, therefore we can
        drop its computation.

        In the present form, the pressure `p` is replaced with `psi`,
        the streamfunction defined on vertices, and `p = f0*psi`.

        """
        addvortexforce(param, mesh, s.U, s.omega, ds.u)
        addcoriolis(param, mesh, s.U, ds.u)
        # addgrad(mesh, s.ke, ds.u)
        # addgrad(mesh, s.p, ds.u)
        divflux(param, mesh, s.flx, s.h, s.U, ds.h)
        qg_projection(mesh, ds.u, ds.h, s.pv, s.psi, anomaly=True)
        fill(mesh, ds.u, ds.h)

    def diag(s):
        sharp(mesh, s.u, s.U)
        compute_vorticity(mesh, s.u, s.omega)
        # compute_kinetic_energy(param, mesh, s.u, s.U, s.ke)
        # compute_pressure(param, mesh, s.h, s.p)
        fill(mesh, s.omega)

    return (rhs, diag)


def get_rsw(param, mesh):

    def rhs(s, ds):
        """ RHS for RSW model"""
        addvortexforce(param, mesh, s.U, s.omega, ds.u)
        addcoriolis(param, mesh, s.U, ds.u)
        addgrad(mesh, s.ke, ds.u)
        addgrad(mesh, s.p, ds.u)
        divflux(param, mesh, s.flx, s.h, s.U, ds.h)
        fill(mesh, ds.u, ds.h)

    def diag(s):
        sharp(mesh, s.u, s.U)
        compute_vorticity(mesh, s.u, s.omega)
        compute_kinetic_energy(param, mesh, s.u, s.U, s.ke)
        compute_pressure(param, mesh, s.h, s.p)
        fill(mesh, s.omega, s.ke)

    return (rhs, diag)


def get_advection(param, mesh):

    def rhs(s, ds):
        """ RHS for Tracer Advection model """
        divflux(param, mesh, s.flx, s.q, s.U, ds.q)
        fill(mesh, ds.q)

    def diag(s):
        pass

    return (rhs, diag)


def get_vectoradv(param, mesh):

    def rhs(s, ds):
        """ RHS for Vector Advection model """
        addvortexforce(param, mesh, s.U, s.omega, ds.v)
        addgrad(mesh, s.q, ds.v)
        fill(mesh, ds.v)

    def diag(s):
        compute_vorticity(mesh, s.v, s.omega)
        compute_kinetic_energy(param, mesh, s.v, s.U, s.q)
        s.q[:] *= 2
        fill(mesh, s.omega, s.q)

    return (rhs, diag)


def get_rhs_and_diag(param, mesh):

    equations = {
        "euler": get_euler,
        "boussinesq": get_boussinesq,
        "hydrostatic": get_hydrostatic,
        "eulerpsi": get_eulerpsi,
        "rsw": get_rsw,
        "qgrsw": get_qgrsw,
        "qg": get_qg,
        "advection": get_advection,
        "vectoradv": get_vectoradv,
    }

    rhs, diag = equations[param.model](param, mesh)

    if param.tracer:
        print("[INFO] add a tracer equation")
        rhs = addtracerequation(param, mesh, rhs, param.tracer)

    if param.forcing:
        print("[INFO] add a forcing term")
        rhs = addforcingterm(param, mesh, rhs, param.forcing)

    return rhs, diag


def addtodocstring(func, rhs, supplementarydoc):
    func.__doc__ = "\n".join([rhs.__doc__, supplementarydoc])


def addtracerequation(param, mesh, rhs, tracername):
    def newrhs(s, ds):
        rhs(s, ds)
        divflux(param, mesh, s.flx,
                getattr(s, tracername), s.U,
                getattr(ds, tracername))

    addtodocstring(newrhs, rhs, f"with tracer {tracername} equation")

    return newrhs


def addforcingterm(param, mesh, rhs, forcing):
    def newrhs(s, ds):
        rhs(s, ds)
        forcing(param, mesh, s, ds)

    addtodocstring(newrhs, rhs, "with forcing term")

    return newrhs
