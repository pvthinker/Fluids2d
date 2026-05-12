import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from .dissipation_diag import Dissipation
from .restart_tools import load_from_restart

def plot_budgets(budget):

    b = budget

    t = b.data["time"]
    dt = np.diff(t)

    K = b.data["KE"]
    A = b.data["APE"]
    B = b.data["BPE"]-b.data["BPE"][1]
    T = K+A+B

    C = b.data["Conv"]
    F = b.data["ForcA"]
    eP = b.data["epsP"]
    eB = b.data["epsB"]
    eK = b.data["epsK"]
    eT = eP-eB

    avg = lambda x: 0.5*(x[1:]+x[:-1])
    ddt = lambda x: np.diff(x)/dt

    tm = avg(t)

    fig, axs = plt.subplots(2,2,figsize=(12,7), sharex=True)
    ax=axs[0,0]
    ax.plot(t, C, label="C")
    ax.plot(t, eK, label="epsK")
    ax.plot(tm, ddt(K), label="dK/dt",lw=3)
    ax.plot(tm, ddt(K)-avg(C+eK), label="dK/dt-C-epsK", lw=3)
    ax.legend(fontsize="xx-small")
    ax.set_title("KE")
    ax.grid()

    ax=axs[0,1]
    ax.plot(t, -C, label="-C")
    ax.plot(t, eP, label="epsP")
    ax.plot(tm, ddt(A), label="dA/dt",lw=3)
    ax.plot(tm, ddt(A)+avg(C-eP-F), label="dA/dt+C-epsP-F", lw=3)
    ax.plot(t, F, label="F")
    ax.legend(fontsize="xx-small")
    ax.set_title("APE")
    ax.grid()

    ax=axs[1,1]
    ax.plot(t, -eB, label="-epsB")
    ax.plot(t, -F, label="-F")
    ax.plot(tm, ddt(B), label="dB/dt",lw=3)
    ax.plot(tm, ddt(B)+avg(eB+F), label="epsB+F+dB/dt",lw=3)
    ax.legend(fontsize="xx-small")
    ax.set_title("BPE")
    ax.set_xlabel("time")
    ax.grid()

    ax=axs[1,0]
    ax.plot(t, eK, label="epsK")
    ax.plot(t, eT, label="epsT")
    ax.plot(tm, ddt(T), label="dT/dt",lw=3)
    ax.plot(tm, ddt(T)-avg(eK+eT), label="dT/dt-epsK-epsT",lw=3)
    ax.legend(fontsize="xx-small")
    ax.set_title("TE")
    ax.set_xlabel("time")
    ax.grid()

    plt.tight_layout()

    fig, ax = plt.subplots()
    ax.plot(t, eK, label="epsK")
    ax.plot(t, eP, label="epsP")
    ax.plot(t, eT, label="epsT")
    ax.plot(t, F, label="F")
    ax.plot(t, C, label="C")
    ax.legend(fontsize="xx-small")
    ax.set_title("Energy Fluxes")
    ax.set_xlabel("time")
    ax.grid()
    plt.tight_layout()

def plot_background_evolution(model, ncfile, n=10):
    dissip = Dissipation(model)
    ds = xr.load_dataset(ncfile)
    nt = len(ds.t)
    idx = np.linspace(0,nt-1,n, dtype="i")
    fig, ax = plt.subplots()
    for k in idx:
        load_from_restart(model,ncfile,k)
        dissip.update_background()
        z = dissip.bp.z1d
        brz = dissip.bp.br(z)
        if k == 0:
            kwargs = {"color":"b", "lw":2, "label":"first"}
        elif k == nt-1:
            kwargs = {"color":"r", "lw":2, "label": "last"}
        else:
            kwargs = {"color":"k", "lw":0.5, "alpha":0.5}

        ax.plot(brz, z, **kwargs)
        print(f"\rprocess ite={k}/{nt}", end="")
    ax.legend()
    ax.set_ylabel("z")
    ax.set_xlabel(r"$b_r$")
    plt.tight_layout()
