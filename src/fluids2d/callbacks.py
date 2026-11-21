import matplotlib.pyplot as plt

from .io import IO
from .movietools import Movie


class Callback:
    def __init__(self, func, name, nite=0, deltat=0,finalize=None):
        self.func = func
        self.name = name
        self.nite = nite
        self.deltat = deltat
        self.lasttime = None
        self.finalize = finalize

    def __call__(self, state, time):
        if self.is_time_to_call(time):
            self.func(state, time)

    def is_time_to_call(self, time):
        if self.nite>0:
            return (time.ite-time.ite0)%self.nite == 0

        if self.lasttime is None:
            self.lasttime = time.t0
        if time.t>=(self.lasttime + self.deltat):
            self.lasttime = time.t
            return True
        else:
            return False

    def __repr__(self):
        freq = f"nite = {self.nite}" if self.nite >0 else f"deltat = {self.deltat:.02f}"
        return f"<{self.name}, {freq}>"


class OnlineMovie:
    def __init__(self, fig, func, **kwargs):
        self.movie = Movie(fig, **kwargs)
        self.func = func
        self.k = 1
        fig.canvas.draw()
        self.movie.addframe()

    def __call__(self, state, time):
        self.func(state, time)
        self.movie.addframe()
        if self.k % 25 == 0:
            plt.pause(1e-6)
        self.k += 1


    def finalize(self):
        print(f"{self.movie.filename} is ready")
        self.movie.finalize()


class History:
    def __init__(self, model, filename="his.nc"):
        self.io = IO(model.param, model.mesh, model.state, model.time, filename)
        self.filename = filename
        self.k = 0
        self.__call__(model.state, model.time)

    def __call__(self, state, time):
        print(f" his={self.k}",end="")
        self.io.write(state,time)
        self.k += 1

    def finalize(self):
        print(f"{self.io.filename} is ready")
