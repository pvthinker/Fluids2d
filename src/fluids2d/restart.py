from netCDF4 import Dataset
import fluids2d as f2d


def get_keys_from_param(param):
    exceptions = ["__parameters__"]
    return [key
            for key in param.__dict__
            if key not in exceptions]


def read_param(ncfile):
    param = f2d.Param()
    keys = get_keys_from_param(param)
    with Dataset(ncfile) as nc:
        for key in keys:
            val = nc.getncattr(key)
            setattr(param, key, val)
    return param


def check_two_params_are_same(param1, param2):
    # pb with "clims" that is a np.array
    keys = get_keys_from_param(param1)
    passed = {key: getattr(param1, key) == getattr(param2, key)
              for key in keys}
    # else
    # key: all(getattr(param1, key) == getattr(param2, key))}
    # if not all(passed.values()):
    #     print(passed)
    return passed


def set_state_and_time_from_file(ncfile, model):
    s = model.state
    t = model.time
    with Dataset(ncfile) as nc:
        s.u.x[:] = nc.variables["ux"][-1]
        s.u.y[:] = nc.variables["uy"][-1]
        t.t = float(nc.variables["t"][-1])
        t.ite = int(nc.variables["ite"][-1])

    t.ite0 = t.ite
    t.t0 = t.t
    model.integrator.diag(s)


def get_model_from_param(ncfile):
    param = read_param(ncfile)
    model = f2d.Model(param)
    return model


def get_model_from_restart(ncfile):
    param = read_param(ncfile)
    model = f2d.Model(param)
    set_state_and_time_from_file(ncfile, model)
    return model
