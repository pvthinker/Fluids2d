import numpy as np
import fluids2d as f2d
from fluids2d.operators import compute_vorticity, verticestocenters, perpgrad
from fluids2d.callbacks import OnlineMovie, History

"""
this script, based on tqg_vortex.py illustrates how to use callbacks.
Here we define 3 callbacks

  - a one panel animation, like the default Fluids2d
  - a two panels animation, with a mp4 file generated
  - a write to his.nc file, a substitute to the regular Fluids2d way of doing it

for the two animations, the figures are defined inside this script, allowing a
complete tuning of the figure

"""

def gaussian(x, y, r): return np.exp(-(x**2+y**2)/(2*r**2))

def compute_Rd2(param):
    return (param.g*param.H)/param.f0**2

def compute_qgpv_from_psi(mesh, pv, psi, U):
    Rd2 = compute_Rd2(mesh.param)
    work = pv[:,:]*0
    perpgrad(mesh, psi, U)
    compute_vorticity(mesh, U, work) # work is on vertices
    work2 = psi*(1/Rd2)*mesh.area # cf 'maindiag' in meshes.py
    verticestocenters(mesh, work-work2, pv) # pv is on centers
    return work2

def set_initial_dipole(model, samesign=False):
    Lx = model.param.Lx
    Ly = model.param.Ly
    x0, y0, r0 = Lx/2, Ly/2, Lx/10

    b0_over_psi0 = 0.6

    xc, yc = model.mesh.xy("c")
    xv, yv = model.mesh.xy("v")

    psi = model.state.psi
    pv = model.state.pv
    b = model.state.b
    U = model.state.U

    eps = 0.05
    initial_profile = lambda x,y: gaussian((1+eps)*(x-x0),(1-eps)*(y-y0),r0)

    psi[:,:] = initial_profile(xv,yv) # at vertices
    psi[:,:] *= model.mesh.area/model.mesh.dx
    # notice the 1/dx term in psi to make pv and b comparable in magnitude
    compute_qgpv_from_psi(model.mesh, pv, psi, U)

    b[:,:] = b0_over_psi0 * initial_profile(xc,yc)*model.mesh.area # at centers


    model.integrator.diag(model.state)

def two_panels(model):
    s = model.state
    coef = 1/model.mesh.area

    # >>> setup the figure

    x,y = model.mesh.xy("c")
    fig,ax = plt.subplots(1,2,sharey=True,figsize=(12.8,7.2))

    im0 = ax[0].pcolormesh(x,y,s.pv*coef,vmin=-2,vmax=2,cmap="RdBu_r")
    contours = []
    def add_contours(doit=True):
        if len(contours)>0:
            contours.pop().remove()
        if doit:
            contours.append(ax[0].contour(x,y,s.psi,10,colors="g"))

    add_contours()
    ax[0].set_title("pv")
    im1 = ax[1].pcolormesh(x,y,s.b*coef,cmap="RdBu_r")
    ax[1].set_title("b")

    ax[0].set_ylabel("y")
    ax[0].set_xlabel("x")
    ax[1].set_xlabel("x")
    ti = fig.suptitle("t=0.00")
    ax[0].set_aspect("equal")
    ax[1].set_aspect("equal")
    plt.tight_layout()

    # figure is complete <<<

    def update_figure(state, time):
        ti.set_text(f"t={time.t:0.2f}")
        add_contours()
        im0.set_array(s.pv*coef)
        im1.set_array(s.b*coef)
        fig.canvas.draw()

    return fig, update_figure

def one_panel(model):

    x,y = model.mesh.xy("c")
    s = model.state
    coef = 1/model.mesh.area

    # >>> setup the figure

    x,y = model.mesh.xy("c")
    fig,ax = plt.subplots(figsize=(12.8,7.2))

    im0 = ax.pcolormesh(x,y,s.pv*coef,vmin=-3,vmax=3,cmap="RdBu_r")

    ax.set_ylabel("y")
    ax.set_xlabel("x")
    ti = ax.set_title("t=0.00")
    ax.set_aspect("equal")
    cb = plt.colorbar(im0)
    cb.set_label("pv")
    plt.tight_layout()

    # figure is complete <<<

    def update_figure(state, time):
        ti.set_text(f"t={time.t:0.2f}")
        im0.set_array(s.pv*coef)
        fig.canvas.draw()
        plt.pause(1e-6)

    return fig, update_figure

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    plt.ion()

    param = f2d.Param()

    param.model = "tqg"
    param.Lx = 1.0
    param.ny = 100
    param.nx = param.ny*int(param.Lx/param.Ly)
    param.tend = 100
    param.dt = 0.1*(200/param.ny)
    param.f0 = 1.
    param.g = 1
    param.H = 1
    param.maxite = 5000
    param.cfl = 0.9
    param.xperiodic = False
    param.noslip = False

    param.nplot = 10
    param.animation = False
    param.generate_mp4 = False
    param.plotvar = "b"

    param.nhis = 0
    param.var_to_store = ["pv", "b"]

    model = f2d.Model(param)

    set_initial_dipole(model)


    # define the  callbacks
    _, anim1 = one_panel(model)
    model.add_callback(anim1,"online animation one panel",nite=10)

    anim2 = OnlineMovie(*two_panels(model),name="tqg_pv_and_b", framerate=60)
    model.add_callback(anim2,"online animation two panels",nite=1)

    # remark we have set param.nhis = 0
    # the regular history.nc is still generated but it will contain
    # only the first and the last snapshots
    model.add_callback(History(model),"write history",deltat=4)

    print(model.interruptions)

    # run the model
    model.run()

