import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .states import vectors
import matplotlib
from .movietools import Movie

fontsize = 16

font = {  # 'family': 'normal',
    # 'weight': 'normal',
    'size': fontsize}

matplotlib.rc('font', **font)
matplotlib.rc('axes', titlesize=fontsize)


class Figure:
    def __init__(self, param, mesh, state, time):
        self.param = param
        self.mesh = mesh
        figsize = (12.8, 7.2)
        self.fig, self.ax = plt.subplots(figsize=figsize)
        if self.param.generate_mp4:
            self.movie = Movie(self.fig)
        x, y = mesh.xy()
        nh = param.halowidth
        x = crop(x, nh)
        y = crop(y, nh)

        self.isvector = is_vector(param.plotvar)

        if self.isvector:
            u, v = self.get_data(state)
            self.quiver = self.ax.quiver(u, v)
        else:
            if param.clims is None:
                self.im = self.ax.pcolormesh(x, y, self.get_data(state),
                                             cmap=param.cmap)
            else:
                vmin, vmax = param.clims
                self.im = self.ax.pcolormesh(x, y, self.get_data(state),
                                             cmap=param.cmap, vmin=vmin, vmax=vmax)
            cb = plt.colorbar(self.im, location="bottom")
            cb.set_label(param.plotvar)

        self.ti = self.ax.set_title(time.tostring())
        self.ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show(block=False)
        if self.param.generate_mp4:
            self.movie.addframe()

    def get_data(self, state):
        return get_data(self.mesh, self.param.plotvar, state)

    def update(self, state, time):
        if self.isvector:
            u, v = self.get_data(state)
            self.quiver.U[:] = u.reshape(-1)
            self.quiver.V[:] = v.reshape(-1)
        else:
            data = self.get_data(state)
            self.im.set_array(data.reshape(-1))

        self.ti.set_text(time.tostring())
        self.fig.canvas.draw()
        plt.pause(1e-6)
        if self.param.generate_mp4:
            self.movie.addframe()


def subsample(u, v, n):
    return u[::n, ::n], v[::n, ::n]


def is_vector(plotvar):
    return (plotvar[0] in vectors) and (len(plotvar) == 1)


def get_data(mesh, plotvar, state):
    coef = get_scaling_factor(mesh, plotvar)
    nh = mesh.param.halowidth
    if plotvar[0] in vectors:
        if len(plotvar) == 1:
            var = getattr(state, plotvar)
            n = int(np.sqrt(var.x.size)//25)
            return subsample(crop(var.x, nh), crop(var.y, nh), n)*coef
        else:
            component = plotvar[1]
            var = getattr(state, plotvar[0])
            array = getattr(var, component)
    else:
        array = getattr(state, plotvar)
    return crop(array, nh)*coef


def crop(array, nh):
    return array[nh:-nh, nh:-nh]


def get_im_and_ti(fig, ax, param, mesh, x, y, state, time):
    if param.clims is None:
        im = ax.pcolormesh(
            x, y, get_data(mesh, param.plotvar, state), cmap=param.cmap)
    else:
        vmin, vmax = param.clims
        im = ax.pcolormesh(x, y, get_data(param.plotvar, state),
                           cmap=param.cmap, vmin=vmin, vmax=vmax)
    ti = ax.set_title(time.tostring())
    addcolorbartosubplot(fig, ax, im)

    return im, ti


def get_scaling_factor(mesh, plotvar):
    coef = 1
    dx, dy = mesh.dx, mesh.dy
    area = dx*dy
    coefs = {
        "omega": 1/area,
        "Ux": dx,
        "Uy": dy,
        "ux": 1/dx,
        "uy": 1/dy,
        "h": 1/area,
        "pv": 1/area,
    }
    if plotvar in coefs:
        coef = coefs[plotvar]

    return coef


class FigureTwin:
    def __init__(self, model1, model2, hstack):
        self.mesh1 = model1.mesh
        self.mesh2 = model2.mesh
        self.param1 = model1.param
        self.param2 = model1.param

        x, y = model1.mesh.xy()
        nh = model1.param.halowidth
        x = crop(x, nh)
        y = crop(y, nh)

        if hstack:
            stack = (1, 2)
        else:
            stack = (2, 1)

        fig, axs = plt.subplots(*stack, figsize=(16, 8))

        param, state, time = model1.param, model1.state, model1.time
        self.im1, self.ti1 = get_im_and_ti(
            fig, axs[0], param, self.mesh1, x, y, state, time)

        param, state, time = model2.param, model2.state, model2.time
        self.im2, self.ti2 = get_im_and_ti(
            fig, axs[1], param, self.mesh2, x, y, state, time)

        self.fig = fig
        self.axs = axs

    def update(self, state1, state2, time):
        data1 = get_data(self.mesh1, self.param1.plotvar, state1)
        data2 = get_data(self.mesh2, self.param2.plotvar, state2)

        self.im1.set_array(data1.reshape(-1))
        self.ti1.set_text(time.tostring())

        self.im2.set_array(data2.reshape(-1))
        self.ti2.set_text(time.tostring())

        self.fig.canvas.draw()
        plt.pause(1e-6)


def addcolorbartosubplot(fig, ax, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='5%', pad=0.3)
    fig.colorbar(im, cax=cax, orientation='horizontal')
