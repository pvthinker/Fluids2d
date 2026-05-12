import numpy as np
import xarray as xr
from fluids2d.states import allocate_var
from fluids2d.operators import tracflux, addvortexforce

from .background_profile import BackgroundProfile


class Dissipation:
    """ Compute the local dissipation rates

    of KE, APE and variance of buoyancy

    along with the BPE injection rate due to APE dissipation
    """
    def __init__(self, model):
        self.model = model
        self.state = model.state
        self.nh = model.param.halowidth
        shape = model.mesh.shape
        self.bflx = allocate_var("flx", shape)
        self.vforce = allocate_var("flx", shape)
        self.epsK = allocate_var("eps", shape)
        self.epsP = allocate_var("eps", shape)
        self.epsB = allocate_var("eps", shape)
        self.epsb2 = allocate_var("eps", shape)

        self.update_background()

    def update_background(self):
        self.bp = BackgroundProfile(crop(self.state.b,nh=self.nh))

    def compute_ape_dissipation(self):
        b = self.state.b
        F = self.bflx
        self.dz = np.zeros_like(b)
        self.zrb = np.zeros_like(b)
        nh = self.nh
        self.zrb[nh:-nh,nh:-nh] = self.bp.zr(crop(b))
        self.dz[nh:-nh,nh:-nh] = crop(self.zrb)-self.bp.z

        compute_innerprod_gradtrac_vect(self.epsP, self.dz, F)
        compute_innerprod_gradtrac_vect(self.epsB, self.zrb, F)

    def compute_b2_dissipation(self):
        b = self.state.b
        F = self.bflx
        compute_innerprod_gradtrac_vect(self.epsb2, b, F)

    def compute_ke_dissipation(self):
        U = self.state.U
        F = self.vforce
        prod = U.x*F.x
        self.epsK[:, :-1] = (prod[:,1:]+prod[:,:-1])*0.5

        prod = U.y*F.y
        self.epsK[:-1, :] += (prod[1:,:]+prod[:-1,:])*0.5

    def compute(self):
        self.update_background()
        state = self.state
        self.irrevtracflux(self.bflx, state.flx, state.b, state.U)
        self.irrevvforce(self.vforce, state.U, state.omega, state.flx)
        self.compute_ke_dissipation()
        self.compute_b2_dissipation()
        self.compute_ape_dissipation()

    def irrevvforce(self, irrevvforce, U, omega, flx):
        set_vector_to_zero(irrevvforce)

        addvortexforce(self.model.param, self.model.mesh, U, omega, flx)
        addtovector(irrevvforce, flx)
        flip(U)

        addvortexforce(self.model.param, self.model.mesh, U, omega, flx)
        addtovector(irrevvforce, flx)
        flip(U)
        irrevvforce.x[:,:] *= 0.5
        irrevvforce.y[:,:] *= 0.5

    def irrevtracflux(self, irrevflx, flx, q, U):
        set_vector_to_zero(irrevflx)

        tracflux(self.model.param, self.model.mesh, flx, q, U)
        addtovector(irrevflx, flx)
        flip(U)

        tracflux(self.model.param, self.model.mesh, flx, q, U)
        addtovector(irrevflx, flx)
        flip(U)
        irrevflx.x[:,:] *= 0.5
        irrevflx.y[:,:] *= 0.5

def addtovector(out, vect):
    out.x[:,:] += vect.x
    out.y[:,:] += vect.y

def set_vector_to_zero(vect):
    vect.x[:,:] = 0
    vect.y[:,:] = 0

def flip(U):
    U.x[:,:] = -U.x
    U.y[:,:] = -U.y

def crop(array,nh=3):
    """ remove the halo from arrray """

    if array.ndim == 1:
        return array[nh:-nh]

    elif array.ndim == 2:
        return array[nh:-nh,nh:-nh]

    elif array.ndim == 3:
        # axis 0 is time -> no halo
        return array[:, nh:-nh,nh:-nh]

def compute_innerprod_gradtrac_vect(out, b, F):
        prod = b*0
        prod[:, 1:] = (b[:,1:]-b[:,:-1])*F.x[:,1:]
        out[:, :-1] = 0.5*(prod[:,1:]+prod[:,:-1])
        prod = b*0
        prod[1:,:] = (b[1:,:]-b[:-1,:])*F.y[1:,:]
        out[:-1,:] += 0.5*(prod[1:,:]+prod[:-1,:])
