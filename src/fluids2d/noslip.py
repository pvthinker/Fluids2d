import numpy as np


def get_slipcoef(param, msk):
    coef = np.zeros(msk.shape)

    coef[:, :] += msk
    coef[:, 1:] += msk[:, :-1]
    coef[1:, :] += msk[:-1, :]
    coef[1:, 1:] += msk[:-1, :-1]

    def freeslip(x): return 1*(x == 4)
    def noslip(x): return np.minimum(x, 1)

    if (param.noslip is None) or (param.noslip == False):
        slipcoef = freeslip(coef)

    elif param.noslip == True:
        slipcoef = noslip(coef)

    else:
        slipcoef = freeslip(coef)
        nh = param.halowidth
        if "left" in param.noslip:
            slipcoef[:, nh] = noslip(coef[:, nh])

        if "right" in param.noslip:
            slipcoef[:, -nh] = noslip(coef[:, -nh])

        if "bottom" in param.noslip:
            slipcoef[nh, :] = noslip(coef[nh, :])

        if "top" in param.noslip:
            slipcoef[-nh, :] = noslip(coef[-nh, :])

    return slipcoef
