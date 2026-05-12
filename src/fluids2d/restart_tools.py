import xarray as xr
from functools import lru_cache

#@lru_cache
def get_ds(ncfile):
    return xr.load_dataset(ncfile)

def load_from_restart(model, ncfile, kt=-1):
    b = model.state.b
    u = model.state.u

    dx = model.mesh.dx
    dy = model.mesh.dy

    ds = get_ds(ncfile)
    b[:,:] = ds.b[kt]
    u.x[:,:] = ds.Ux[kt]*dx**2
    u.y[:,:] = ds.Uy[kt]*dy**2

    model.time.t = float(ds.t[kt].data)

    model.integrator.diag(model.state)
