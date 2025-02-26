import numpy as np
import signal
from time import time
from .meshes import Mesh
from .states import State
from .integrators import get_integrator
from .timeline import Time
from .animation import Figure
from .io import IO
from .equations import addforcingterm


class Model:
    def __init__(self, param):
        param.check()
        self.param = param

        self.mesh = Mesh(param)
        self.state = State(param, self.mesh.shape)
        self.set_integrator()
        self.time = Time(param)
        self.io = IO(param, self.mesh, self.state, self.time)
        self.callbacks = []

    def set_integrator(self):
        self.integrator = get_integrator(self.param, self.mesh, self.state)

    def run(self):
        if self.param.animation:
            self.execute_callbacks()
            self.figure = Figure(self.param, self.mesh, self.state, self.time)
        self.stop = False

        def signal_handler(signal, frame):
            print('\n hit ctrl-C, stopping', end='')
            self.stop = True

        signal.signal(signal.SIGINT, signal_handler)

        tic = time()

        self.save_to_file()

        while (not self.time.finished) and (not self.stop):
            self.set_dt()
            self.step(1)
            self.progress()
            self.animation()
            self.save_to_file()

        self.progress()
        toc = time()

        elapsed = toc-tic
        self.print_perf(elapsed)

        if self.param.generate_mp4:
            self.figure.movie.finalize()

    def step(self, nsteps=1):
        for _ in range(nsteps):
            self.integrator.step(self.state, self.time)

    def set_dt(self):
        if self.param.dt > 0:
            self.time.dt = self.param.dt
            return

        if self.param.model == "rsw":
            g = self.param.g
            H = self.param.H
            c = (g*H)**0.5
            U = c/self.mesh.dx
            V = c/self.mesh.dy
            maxU = U+V
        else:
            U = self.state.U
            maxU = np.max(np.abs(U.x))+np.max(np.abs(U.y))+1e-99

        self.time.dt = min(self.param.cfl/maxU, self.param.dtmax)

    def print_perf(self, elapsed):
        print()
        perf = elapsed/(self.mesh.nx*self.mesh.ny*self.time.ite)
        print(f"Elapsed: {elapsed:.2f} s  perf: {perf:.2e} s/dof")

    def progress(self):
        if (self.time.ite % self.param.nprint == 0) or (self.time.finished):

            msg = [f"\rite={self.time.ite}",
                   f"{self.time.tostring()}",
                   f"dt={self.time.dt:.2g}"]

            print(" ".join(msg), end="")

    def animation(self):
        if self.time.update_anim:
            self.execute_callbacks()
            self.figure.update(self.state, self.time)

    def save_to_file(self):
        if self.time.save_to_file:
            self.io.write(self.state, self.time)

    def execute_callbacks(self):
        for func in self.callbacks:
            func(self.param, self.mesh, self.state, self.time)

    def add_forcing(self, forcing):
        self.integrator.rhs = addforcingterm(
            self.param, self.mesh, self.integrator.rhs, forcing)
