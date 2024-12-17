import numpy as np
from numba import njit, jit
from .parallel import thread_execution

eps = 1e-40
akima_threshold = np.vectorize(lambda x, y: 2*x*y/(x+y) if x*y > eps else 0)


def flux_1d_akima(flx, U, q):

    dq = flx
    dq[1:-1] = q[1:]-q[:-1]
    dq[0] = dq[1]
    dq[-1] = dq[-2]
    cf = akima_threshold(dq[1:], dq[:-1])

    flx[1:-1] = U[1:-1]*0.5*(q[1:]+q[:-1] - (cf[1:]-cf[:-1])*(1/3))
    flx[[0, -1]] = 0


@njit("f8(f8,f8,f8)")
def ce2(U, qm, qp):
    return (qm+qp)*0.5


@njit("f8(f8,f8,f8,f8,f8)")
def ce4(U, qmm, qm, qp, qpp):
    return (-qmm+7*(qm+qp)-qpp)/12


@njit("f8(f8,f8,f8,f8,f8,f8,f8)")
def ce6(U, qmmm, qmm, qm, qp, qpp, qppp):
    return (qmmm-8*(qmm+qpp)+37*(qm+qp)+qppp)/60


@njit("f8(f8,f8,f8)")
def up3(qm, q0, qp):
    return (5*q0+2*qp-qm)/6


@njit("f8(f8,f8,f8,f8,f8)")
def up5(qmmm, qmm, qm, qp, qpp):
    return (2*qmmm - 13*qmm + 47*qm + 27*qp - 3*qpp)/60


@njit("f8(f8,f8,f8)")
def weno3(qm, q0, qp):
    # reconstruction of qhalf, mid-point between q0 and qp
    # with qm on the upwind side (where the flow comes from)
    #
    # if the flow speed is positive
    # qhalf = weno3(q[i-1],q[i],q[i+1])
    #
    # if the flow speed is negative
    # qhalf = weno3(q[i+2],q[i+1],q[i])
    #

    eps = 1e-8  # <- to avoid the division by zero
    # WATCH OUT: the value is appropriate for q ~ O(1)
    #
    # For q of O(10^n) then use eps = 10**(n-8)

    beta_1 = (q0-qm)**2
    beta_2 = (qp-q0)**2

    w_1 = 1/(beta_1+eps)**2
    w_2 = 2/(beta_2+eps)**2

    P1 = 1.5*q0-0.5*qm
    P2 = 0.5*qp+0.5*q0

    return (w_1*P1 + w_2*P2)/(w_1+w_2)


@njit("f8(f8,f8,f8)")
def weno3z(qm, q0, qp):
    """
    3-points non-linear left-biased stencil reconstruction:

    qm-----q0--x--qp

    An improved weighted essentially non-oscillatory scheme for hyperbolic
    conservation laws, Borges et al, Journal of Computational Physics 227 (2008).
    """
    eps = 1e-14

    qi1 = -1./2.*qm + 3./2.*q0
    qi2 = 1./2.*(q0 + qp)

    beta1 = (q0-qm)**2
    beta2 = (qp-q0)**2
    tau = np.abs(beta2-beta1)

    g1, g2 = 1./3., 2./3.
    w1 = g1 * (1. + tau / (beta1 + eps))
    w2 = g2 * (1. + tau / (beta2 + eps))

    qi_weno3 = (w1*qi1 + w2*qi2) / (w1 + w2)

    return qi_weno3


@njit("f8(f8, f8,f8, f8, f8)")
def cweno3z(U, qmm, qm, qp, qpp):
    """
    3-points non-linear left-biased stencil reconstruction:

    qm-----q0--x--qp

    An improved weighted essentially non-oscillatory scheme for hyperbolic
    conservation laws, Borges et al, Journal of Computational Physics 227 (2008).
    """
    eps = 1e-14

    qi1 = -1./2.*qmm + 3./2.*qm
    qi2 = 1./2.*(qm + qp)
    qi3 = -1./2.*qpp + 3./2.*qp

    if U > 0:
        beta1 = (qm-qmm)**2
        beta2 = (qp-qm)**2
    else:
        beta1 = (qp-qpp)**2
        beta2 = (qp-qm)**2

    tau = np.abs(beta2-beta1)

    g1, g2 = 1./3., 2./3.
    w1 = g1 * (1. + tau / (beta1 + eps))
    w2 = g2 * (1. + tau / (beta2 + eps))

    qi_weno3 = (w1*(qi1+qi3)*0.5 + w2*qi2) / ((w1 + w2))

    return qi_weno3


@njit("f8(f8,f8,f8,f8,f8)")
def weno5(qmm, qm, q0, qp, qpp):
    """
    5-points non-linear left-biased stencil reconstruction

    qmm----qm-----q0--x--qp----qpp

    Efficient Implementation of Weighted ENO Schemes, Jiang and Shu,
    Journal of Computation Physics 126, 202–228 (1996)
    """
    eps = 1e-8
    qi1 = 1./3.*qmm - 7./6.*qm + 11./6.*q0
    qi2 = -1./6.*qm + 5./6.*q0 + 1./3.*qp
    qi3 = 1./3.*q0 + 5./6.*qp - 1./6.*qpp

    k1, k2 = 13./12., 0.25
    beta1 = k1 * (qmm-2*qm+q0)**2 + k2 * (qmm-4*qm+3*q0)**2
    beta2 = k1 * (qm-2*q0+qp)**2 + k2 * (qm-qp)**2
    beta3 = k1 * (q0-2*qp+qpp)**2 + k2 * (3*q0-4*qp+qpp)**2

    g1, g2, g3 = 0.1, 0.6, 0.3
    w1 = g1 / (beta1+eps)**2
    w2 = g2 / (beta2+eps)**2
    w3 = g3 / (beta3+eps)**2

    qi_weno5 = (w1*qi1 + w2*qi2 + w3*qi3) / (w1 + w2 + w3)

    return qi_weno5


@njit("f8(f8,f8,f8,f8,f8)")
def weno5z(qmm, qm, q0, qp, qpp):
    """
    5-points non-linear left-biased stencil reconstruction

    qmm----qm-----q0--x--qp----qpp

    An improved weighted essentially non-oscillatory scheme for hyperbolic
    conservation laws, Borges et al, Journal of Computational Physics 227 (2008)
    """
    eps = 1e-16

    qi1 = 1./3.*qmm - 7./6.*qm + 11./6.*q0
    qi2 = -1./6.*qm + 5./6.*q0 + 1./3.*qp
    qi3 = 1./3.*q0 + 5./6.*qp - 1./6.*qpp

    k1, k2 = 13./12., 0.25
    beta1 = k1 * (qmm-2*qm+q0)**2 + k2 * (qmm-4*qm+3*q0)**2
    beta2 = k1 * (qm-2*q0+qp)**2 + k2 * (qm-qp)**2
    beta3 = k1 * (q0-2*qp+qpp)**2 + k2 * (3*q0-4*qp+qpp)**2

    tau5 = np.abs(beta1 - beta3)

    g1, g2, g3 = 0.1, 0.6, 0.3
    w1 = g1 * (1 + tau5 / (beta1 + eps))
    w2 = g2 * (1 + tau5 / (beta2 + eps))
    w3 = g3 * (1 + tau5 / (beta3 + eps))

    qi_weno5 = (w1*qi1 + w2*qi2 + w3*qi3) / (w1 + w2 + w3)

    return qi_weno5


@njit("f8(f8,f8,f8,f8,f8,f8,f8)")
def cweno5z(U, qmmm, qmm, qm, qp, qpp, qppp):
    """
    5-points non-linear left-biased stencil reconstruction

    qmm----qm-----q0--x--qp----qpp

    An improved weighted essentially non-oscillatory scheme for hyperbolic
    conservation laws, Borges et al, Journal of Computational Physics 227 (2008)
    """
    eps = 1e-16

    qi1 = 1./3.*qmmm - 7./6.*qmm + 11./6.*qm
    qi2 = -1./6.*qmm + 5./6.*qm + 1./3.*qp
    qi3 = 1./3.*qm + 5./6.*qp - 1./6.*qpp

    qi4 = 1./3.*qp + 5./6.*qm - 1./6.*qmm
    qi5 = -1./6.*qpp + 5./6.*qp + 1./3.*qm
    qi6 = 1./3.*qppp - 7./6.*qpp + 11./6.*qp

    k1, k2 = 13./12., 0.25

    beta1 = k1 * (qmmm-2*qmm+qm)**2 + k2 * (qmmm-4*qmm+3*qm)**2
    beta2 = k1 * (qmm-2*qm+qp)**2 + k2 * (qmm-qp)**2
    beta3 = k1 * (qm-2*qp+qpp)**2 + k2 * (3*qm-4*qp+qpp)**2

    beta6 = k1 * (qmmm-2*qmm+qm)**2 + k2 * (qmmm-4*qmm+3*qm)**2
    beta5 = k1 * (qmm-2*qm+qp)**2 + k2 * (qmm-qp)**2
    beta4 = k1 * (qm-2*qp+qpp)**2 + k2 * (3*qm-4*qp+qpp)**2

    tau5p = np.abs(beta1 - beta3)
    tau5m = np.abs(beta6 - beta4)

    g1, g2, g3 = 0.1, 0.6, 0.3

    w1 = g1 * (1 + tau5p / (beta1 + eps))
    w2 = g2 * (1 + tau5p / (beta2 + eps))
    w3 = g3 * (1 + tau5p / (beta3 + eps))

    w6 = g1 * (1 + tau5m / (beta6 + eps))
    w5 = g2 * (1 + tau5m / (beta5 + eps))
    w4 = g3 * (1 + tau5m / (beta4 + eps))

    # if U>0:
    #     qrec = (w1*qi1 + w2*qi2 + w3*qi3) / (w1 + w2 + w3)
    # else:
    #     qrec = (w6*qi6 + w5*qi5 + w4*qi4) / (w6 + w5 + w4)

    qp = (w1*qi1 + w2*qi2 + w3*qi3) / (w1 + w2 + w3)
    qm = (w6*qi6 + w5*qi5 + w4*qi4) / (w4 + w5 + w6)
    qrec = (qp+qm)*0.5

    return qrec


@njit("f8(f8,f8,f8,f8,f8,f8,f8)")
def cweno5z_v0(U, qmmm, qmm, qm, qp, qpp, qppp):
    """
    5-points non-linear left-biased stencil reconstruction

    qmm----qm-----q0--x--qp----qpp

    An improved weighted essentially non-oscillatory scheme for hyperbolic
    conservation laws, Borges et al, Journal of Computational Physics 227 (2008)
    """
    eps = 1e-16

    qi1 = 1./3.*qmmm - 7./6.*qmm + 11./6.*qm
    qi2 = -1./6.*qmm + 5./6.*qm + 1./3.*qp
    qi3 = 1./3.*qm + 5./6.*qp - 1./6.*qpp

    qi4 = 1./3.*qp + 5./6.*qm - 1./6.*qmm
    qi5 = -1./6.*qpp + 5./6.*qp + 1./3.*qm
    qi6 = 1./3.*qppp - 7./6.*qpp + 11./6.*qp

    k1, k2 = 13./12., 0.25
    if U > 0:
        beta1 = k1 * (qmmm-2*qmm+qm)**2 + k2 * (qmmm-4*qmm+3*qm)**2
        beta2 = k1 * (qmm-2*qm+qp)**2 + k2 * (qmm-qp)**2
        beta3 = k1 * (qm-2*qp+qpp)**2 + k2 * (3*qm-4*qp+qpp)**2
    else:
        beta1 = k1 * (qmmm-2*qmm+qm)**2 + k2 * (qmmm-4*qmm+3*qm)**2
        beta2 = k1 * (qmm-2*qm+qp)**2 + k2 * (qmm-qp)**2
        beta3 = k1 * (qm-2*qp+qpp)**2 + k2 * (3*qm-4*qp+qpp)**2

    tau5 = np.abs(beta1 - beta3)

    g1, g2, g3 = 0.1, 0.6, 0.3

    w1 = g1 * (1 + tau5 / (beta1 + eps))
    w2 = g2 * (1 + tau5 / (beta2 + eps))
    w3 = g3 * (1 + tau5 / (beta3 + eps))

    # if U>0:
    #     qrec = (w1*qi1 + w2*qi2 + w3*qi3) / (w1 + w2 + w3)
    # else:
    #     qrec = (w1*qi6 + w2*qi5 + w3*qi4) / (w1 + w2 + w3)

    qrec = (w1*(qi1+qi6) + w2*(qi2+qi5) + w3*(qi3+qi4)) / (2*(w1 + w2 + w3))

    return qrec


@njit("f8(f8,f8,f8)")
def flx1(U, qm, qp):
    return qm if U > 0 else qp


@njit("f8(f8,f8,f8,f8,f8)")
def flx3(U, qmm, qm, qp, qpp):
    return weno3z(qmm, qm, qp) if U > 0 else weno3z(qpp, qp, qm)


@njit("f8(f8,f8,f8,f8,f8)")
def cflx3(U, qmm, qm, qp, qpp):
    return cweno3z(U, qmm, qm, qp, qpp)


@njit("f8(f8,f8,f8,f8,f8)")
def flxup3(U, qmm, qm, qp, qpp):
    return up3(qmm, qm, qp) if U > 0 else up3(qpp, qp, qm)


@njit("f8(f8,f8,f8,f8,f8,f8,f8)")
def flx5(U, qmmm, qmm, qm, qp, qpp, qppp):
    return weno5z(qmmm, qmm, qm, qp, qpp) if U > 0 else weno5z(qppp, qpp, qp, qm, qmm)


@njit("f8(f8,f8,f8,f8,f8,f8,f8)")
def cflx5(U, qmmm, qmm, qm, qp, qpp, qppp):
    return cweno5z_v0(U, qmmm, qmm, qm, qp, qpp, qppp)


@njit("f8(f8,f8,f8,f8,f8,f8,f8)")
def flxup5(U, qmmm, qmm, qm, qp, qpp, qppp):
    return up5(qmmm, qmm, qm, qp, qpp) if U > 0 else up5(qppp, qpp, qp, qm, qmm)


_fluxes = {
    "weno": (flx1, flx3, flx5),
    "upwind": (flx1, flxup3, flxup5),
    "centered": (ce2, ce4, ce6),
    "cweno": (ce2, cflx3, cflx5),
}


def get_compflux_on_interval(method):
    f1, f3, f5 = _fluxes[method]

    @njit("void(f8[:],f8[:],f8[:],i1[:],i4,i4,i4)")
    def compflux_on_interval(flx, U, q, o, s, i0, i1):
        for i in range(i0, i1):
            if o[i] > 4:
                flx[i] = f5(U[i], q[i-3*s], q[i-2*s],
                            q[i-s], q[i],
                            q[i+s], q[i+2*s])*U[i]
            elif o[i] > 2:
                flx[i] = f3(U[i], q[i-2*s],
                            q[i-s], q[i],
                            q[i+s])*U[i]
            elif o[i] > 0:
                flx[i] = f1(U[i], q[i-s], q[i])*U[i]
            else:
                flx[i] = 0
    return compflux_on_interval


def get_vortexforce_on_interval(method):
    f1, f3, f5 = _fluxes[method]

    @njit("void(f8[:],f8[:],f8[:],i1[:],i4,i4,i4,i4,i4)")
    def vortexforce_on_interval(du, V, q, o, s, s2, sign, i0, i1):
        for i in range(i0, i1):
            if o[i] > 4:
                Vm = 0.25*(V[i]+V[i+s]+V[i-s2]+V[i+s-s2])
                du[i] = sign*f5(Vm, q[i-2*s], q[i-s], q[i],
                                q[i+s], q[i+2*s], q[i+3*s])*Vm
            elif o[i] > 2:
                Vm = 0.25*(V[i]+V[i+s]+V[i-s2]+V[i+s-s2])
                du[i] = sign*f3(Vm, q[i-s], q[i], q[i+s], q[i+2*s])*Vm
            elif o[i] > 0:
                Vm = 0.25*(V[i]+V[i+s]+V[i-s2]+V[i+s-s2])
                du[i] = sign*f1(Vm, q[i], q[i+s])*Vm
            else:
                du[i] = 0
    return vortexforce_on_interval


def get_innerproduct_on_interval(method):
    f1, f3, f5 = _fluxes[method]

    @njit("void(f8[:],f8[:],f8[:],i1[:],i4,i4,i4)")
    def innerproduct_on_interval(ke, U, q, o, s, i0, i1):
        for i in range(i0, i1):
            if o[i] > 4:
                Um = 0.5*(U[i]+U[i+s])
                ke[i] += f5(Um, q[i-2*s], q[i-s], q[i],
                            q[i+s], q[i+2*s], q[i+3*s])*Um
            elif o[i] > 2:
                Um = 0.5*(U[i]+U[i+s])
                ke[i] += f3(Um, q[i-s], q[i], q[i+s], q[i+2*s])*Um
            elif o[i] > 0:
                Um = 0.5*(U[i]+U[i+s])
                ke[i] += f1(Um, q[i], q[i+s])*Um

    return innerproduct_on_interval


# flatten operator
def f(x): return x.reshape(-1)


def compflux(flx, U, q, o, s, funcname, nthreads=1):
    func = CompFlux[funcname]
    if nthreads == 1:
        func(f(flx), f(U), f(q), f(o), s, 0, q.size)
    else:
        thread_execution(func,
                         (f(flx), f(U), f(q), f(o)),
                         (s,),
                         nthreads, 0)


def vortexforce(du, V, omega, o, s, s2, sign, funcname, nthreads=1):
    func = VortexForce[funcname]
    if nthreads > 1:
        thread_execution(func,
                         (f(du), f(V), f(omega), f(o)),
                         (s, s2, sign),
                         nthreads, 0)
    else:
        func(f(du), f(V), f(omega), f(o), s, s2, sign, 0, du.size)


def innerproduct(ke, U, u, o, s, funcname, nthreads=1):
    func = InnerProduct[funcname]
    func(f(ke), f(U), f(u), f(o), s, 0, ke.size)


CompFlux = {method: get_compflux_on_interval(method)
            for method in _fluxes}

VortexForce = {method: get_vortexforce_on_interval(method)
               for method in _fluxes}

InnerProduct = {method: get_innerproduct_on_interval(method)
                for method in _fluxes}
