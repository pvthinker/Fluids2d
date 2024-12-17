import numpy as np
from collections import namedtuple
from .states import Prognostic
from .equations import get_rhs_and_diag

_specs = namedtuple("specs", ("caller", "nstages"))


def get_integrator(param, mesh, state):
    if param.integrator in RKintegrators:
        return RKIntegrator(param, mesh, state)

    elif param.integrator == "LFRA":
        return LFRAintegrator(param, mesh, state)

    else:
        raise NotImplementedError(f"{param.integrator} is not implemented")


class LFRAintegrator:
    """Leap-Frog integrator combined with a R. Asselin filtexr

    The first iteration is done with a Euler forward stepping.

    """

    def __init__(self, param, mesh, state):
        self.scratch = [Prognostic(param, mesh.shape) for _ in range(3)]
        self.RAgamma = param.RAgamma
        self.rhs, self.diag = get_rhs_and_diag(param, mesh)

    def step(self, state, time):
        first = (time.ite == 0)
        LFRA(state, time.t, time.dt, self.rhs, self.diag,
             self.scratch, first=first, gamma=self.RAgamma)
        time.pushforward()


def LFRA(s, t, dt, rhs, diag, scratch, first=False, gamma=0.1):
    sb, sa, ds = scratch

    rhs(s, ds)

    if first:
        copyto(s, sb)
        copyto(s, sa)
        addto(s, dt, ds)
    else:
        addto(sa, 2*dt, ds)
        addto(s, gamma, sa, gamma, sb, -2*gamma, s)
        rightpermute(sa, s, sb)

    diag(s)


rkdocs = {
    "rk3": "Strongly Stably Preserving RK3 (Shu 2003)",
    "ef": "Euler Forward",
    "enrk3": "Energy-Preserving RK3 (Celledoni et al. 2009)"
}


class RKIntegrator:
    """Runge-Kutta integrator"""

    def __init__(self, param, mesh, state):
        specs = RKintegrators[param.integrator]

        self.integrator = specs.caller
        self.scratch = [Prognostic(param, mesh.shape)
                        for _ in range(specs.nstages)]

        self.rhs, self.diag = get_rhs_and_diag(param, mesh)
        self.__doc__ += f"\n{rkdocs[param.integrator]}"

    def step(self, state, time):
        self.integrator(state, time.t, time.dt, self.rhs,
                        self.diag, self.scratch)
        time.pushforward()


def ef(s, t, dt, rhs, diag, scratch):
    """update s with one iteration of Euler forward"""

    ds1, = scratch

    rhs(s, ds1)
    addto(s, dt, ds1)
    diag(s)


def rk3(s, t, dt, rhs, diag, scratch):
    """update s with one iteration of SSP RK3"""

    ds1, ds2, ds3 = scratch

    rhs(s, ds1)
    addto(s, dt, ds1)
    diag(s)

    rhs(s, ds2)
    addto(s, -3*dt/4, ds1, dt/4, ds2)
    diag(s)

    rhs(s, ds3)
    addto(s, -dt/12, ds1, -dt/12, ds2, 2*dt/3, ds3)
    diag(s)

def enrk3(s, t, dt, rhs, diag, scratch):
    """update s with one iteration of Energy-Conserving RK3"""

    ds1, ds2, ds3 = scratch

    rhs(s, ds1)
    addto(s, dt/3, ds1)
    diag(s)

    rhs(s, ds2)
    addto(s, -dt/3 -5*dt/48, ds1, 15*dt/16, ds2)
    diag(s)

    rhs(s, ds3)
    addto(s, 5*dt/48+dt/10, ds1, -7*dt/16, ds2, 2*dt/5, ds3)
    diag(s)


RKintegrators = {
    "rk3": _specs(rk3, 3),
    "ef": _specs(ef, 1),
    "enrk3": _specs(enrk3, 3)
}


def rightpermute(a, b, c):
    copyto(b, c)
    copyto(a, b)
    copyto(c, a)


def copyto(x, y):
    """
    copy y to x
    """
    if hasattr(y, "_fields"):
        # y is a nestedtuple and x is a list of nestedtuple

        for k in range(min(len(y), len(x))):
            copyto(x[k], y[k])
    else:
        assert isinstance(y, np.ndarray)
        y[:] = x[:]


def addto_list(y, coefs, x):
    """y += sum_i coefs[i]*x[i]

    - coefs and x are lists
    - x's are either
      - np.array
      - namedtuple of np.array
      - namedtuple of namedtuple of np.array
      - deeper nesting of namedtuple
    - z has to be mutable

    """
    if hasattr(y, "_fields"):
        # y is a nestedtuple and x is a list of nestedtuple

        for k in range(len(x[0])):
            addto_list(y[k], coefs, [z[k] for z in x])
    else:
        assert isinstance(y, np.ndarray)
        # y is a np.array and x is a list of np.array
        y[:] += sum((c*xx for c, xx in zip(coefs, x)))


def addto(y, *args):
    """addto with a more flexible API than addto_list

    instead of giving coefs and x's as list, they are given in an
    alternate sequence of arbitrary length

    addto(y, c0, x0, c1, x1, c2, x2)

    is equivalent to

    addto_list(y, [c0, c1, c2], [x0, x1, x2])

    but
    addto(y, c0, x0)
    addto(y, c0, x0, c1, x1)
    also work

    """
    assert len(args) % 2 == 0
    addto_list(y, args[::2], args[1::2])
