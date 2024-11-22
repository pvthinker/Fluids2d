from netCDF4 import Dataset
from .states import vectors


class IO:
    def __init__(self, param, mesh, state, time):
        self.param = param
        self.kt = 0
        if param.nhis > 0:
            create_file(param, mesh, state, time)

    def write(self, state, time):
        with Dataset(self.param.outputfile, "r+") as nc:

            nc.variables["t"][self.kt] = time.t
            nc.variables["ite"][self.kt] = time.ite
            nc.variables["dt"][self.kt] = time.dt

            for varname in self.param.var_to_store:
                store_in_netCDF(nc, self.kt, state, varname)

        self.kt += 1


def store_in_netCDF(nc, kt, state, varname):
    if varname in vectors:
        vx, vy = get_data_from_state(state, varname)
        nc.variables[varname+"x"][kt, :, :] = vx
        nc.variables[varname+"y"][kt, :, :] = vy
    else:
        v = get_data_from_state(state, varname)
        nc.variables[varname][kt, :, :] = v


def get_data_from_state(state, varname):
    v = getattr(state, varname)
    if varname in vectors:
        return (v.x, v.y)
    else:
        return v


def create_file(param, mesh, state, time):

    def booltointeger(v): return int(v) if isinstance(v, bool) else v
    def nonetostring(v): return "None" if v is None else v

    def cleanup(dic):
        for k, v in dic.items():
            dic[k] = nonetostring(booltointeger(v))
        return dic

    atts = cleanup(param.__dict__.copy())

    dimensions = {"x": mesh.shape[1],
                  "y": mesh.shape[0],
                  "t": None}

    timevar = ["t", "ite", "dt"]

    dtype = "f"

    with Dataset(param.outputfile, "w", format='NETCDF4') as nc:
        nc.setncatts(atts)

        for dim, size in dimensions.items():
            nc.createDimension(dim, size)

        for varname in ["x", "y"]:
            v = nc.createVariable(varname, dtype, ("y", "x"))

        for varname in timevar:
            dtype = "i4" if varname == "ite" else "f"
            v = nc.createVariable(varname, dtype, ("t",))

        for varname in param.var_to_store:
            if varname in vectors:
                v = nc.createVariable(varname+"x", dtype, ("t", "y", "x"))
                v.standard_name = varname+"x"
                v = nc.createVariable(varname+"y", dtype, ("t", "y", "x"))
                v.standard_name = varname+"y"
            else:
                v = nc.createVariable(varname, dtype, ("t", "y", "x"))
                v.standard_name = varname

        x, y = mesh.xy()
        nc.variables["x"][:, :] = x
        nc.variables["y"][:, :] = y
