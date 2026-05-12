import numpy as np
from scipy import interpolate
from scipy import integrate


def crop(array,nh=3):
    """ remove the halo from arrray """

    if array.ndim == 1:
        return array[nh:-nh]

    elif array.ndim == 2:
        return array[nh:-nh,nh:-nh]

    elif array.ndim == 3:
        # axis 0 is time -> no halo
        return array[:, nh:-nh,nh:-nh]

def subsample(array, nsamples):
    """ extract n equidistributed values from an array

    it includes the first and the entry of the array"""
    assert array.ndim == 1
    idx = np.linspace(0,array.size-1,nsamples,dtype="i")
    return array[idx]

class BackgroundProfile:
    """ Background profile associated with the buoyancy field b

    contains

    zr(b) : function that returns the reference depth for buoyancy b
    br(z) : function that returns the reference buoyancy for depth z
    pr(z) : function that returns the antiderivative of br(z)
    N2r(z): function that returns the reference Brunt Vaisala frequency at depth z
    bval  : sorted vector of b values
    zval  : vector of monotonically increasing depth, corresponding to bval
    z     : two dimensional array of depth, corresponding to depth of b array
    """
    def __init__(self, b, H=1):

        if b.ndim == 2:
            ny,nx = b.shape
            n = nx*4
        elif b.ndim == 3:
            nt, ny, nx = b.shape
            n = nt*nx*4

        dy = 1/ny
        self.idx = np.argsort(b.ravel())
        self.bval = b.ravel()[self.idx]
        self.zval= (np.arange(ny*nx)+0.5)*dy/nx
        #self.zval = np.linspace(dy/nx/2,H-(dy/nx)/2,len(self.bval))

        self.z1d=(np.arange(ny)+0.5)*dy
        self.z=self.z1d[:,np.newaxis]*np.ones((ny,nx))
        #dy = 1/ny
        #z1d = (np.arange(ny)+0.5)*dy
        #self.z=z1d[:,np.newaxis]*np.ones((ny,nx))
        #self.zval = self.z.ravel()[idx]

        bcontrol = subsample(self.bval, n)
        zcontrol = subsample(self.zval, n)
        self.zr=interpolate.interp1d(bcontrol, zcontrol,
                                     kind="slinear",
                                     fill_value="extrapolate")
        self.br=interpolate.interp1d(zcontrol, bcontrol,
                                     kind="slinear",
                                     fill_value="extrapolate")
        eps = H*1e-2
        self.N2r = lambda z : (self.br(z+eps)-self.br(z-eps))/(2*eps)

        pval = np.zeros(len(self.zval)+1)
        pval[1:] = np.cumsum(self.bval*dy/nx)

        zcontrol = np.arange(ny*4+1)*dy/2
        pcontrol = subsample(pval, len(zcontrol))

        self.pr=interpolate.interp1d(zcontrol, pcontrol,
                                     kind="cubic",
                                     fill_value="extrapolate")


    def deltap(self, z, zrb):
        return self.pr(z) - self.pr(zrb)

    def ape(self, b, z):
        """ compute the exact APE density

        b: ndarray of buyancy
        z: ndarray of depth (corresponding to b)

        return
        ape: ndarray of APE density
        """
        zrb = self.zr(b)
        deltap = self.deltap(z, zrb)
        #dz=self.zval[self.idx]-self.z.ravel()[self.idx]
        return deltap + b*(zrb-z)
        #return deltap + b*dz.reshape(b.shape)
