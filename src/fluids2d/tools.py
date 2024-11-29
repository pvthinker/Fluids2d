import matplotlib.pyplot as plt
from .operators import perpgrad
from .animation import FigureTwin, get_data


def set_uv_from_omega(model, omega, u, contravariant=False):
    r"""Compute velocity `u` from a distribution of vorticity `omega`
    using

    .. math::
    u = \nabla^\perp \psi

    and

    .. math::
    \psi = \nabla^2 \omega

    Note: `omega` should be defined on vertices

    """
    mesh = model.mesh

    psi = omega*0

    mesh.poisson_vertices.solve(omega, psi)
    perpgrad(mesh, psi, u, contravariant=contravariant)


def run_twin_experiments(model1, model2):
    """Integrate two models synchronously

    using the same time step (model1.time.dt)

    """
    figure = FigureTwin(model1, model2)

    while not model1.time.finished:
        model1.set_dt()
        model2.time.dt = model1.time.dt

        model1.step(1)
        model2.step(1)

        model1.progress()
        if model1.time.update_anim:
            figure.update(model1.state, model2.state, model1.time)

    model1.progress()
    figure.update(model1.state, model2.state, model1.time)


def browse(model, varname):
    """Quickly browse a model state variable

    Example:

    >>> browse(model, "ux")

    plot model.state.u.x

    """
    array = get_data(varname, model.state)
    plt.clf()
    plt.pcolormesh(array[:-1, :-1])
    plt.title(varname)
    plt.colorbar()
