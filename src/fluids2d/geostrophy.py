
"""
Set the velocity field in geostrophic balance,
or a higher balance, from the mass field


"""
import numpy as np
from fluids2d.operators import centerstovertices, perpgrad, compute_pressure

def set_balance(model, nite=1):
    assert model.param.model == "rsw"
    mesh = model.mesh
    state = model.state
    param = model.param
    g = param.g
    H = param.H
    f0 = param.f0
    area = mesh.area

    h = state.h
    p = state.p
    ke = state.ke
    u = state.u
    psi = np.zeros_like(h)

    assert f0>0
    for kt in range(nite):
        compute_pressure(param, mesh, h, p)
        centerstovertices(mesh, p+ke, psi, addto=False)
        psi[psi==0] = g*H
        psi *= (1/f0)
        perpgrad(mesh, psi, u, contravariant=False)
        model.integrator.diag(state)
    return psi
