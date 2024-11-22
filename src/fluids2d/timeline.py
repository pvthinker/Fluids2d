import numpy as np


class Time:
    def __init__(self, param):
        self.t = 0.0
        self.dt = 0.01
        self.ite = 0
        self.param = param

    @property
    def finished(self):
        return (self.t >= self.param.tend) or (self.ite >= self.param.maxite)

    def pushforward(self):
        self.t += self.dt
        self.ite += 1

    def tostring(self):
        return f"t={self.t:.2f}"

    @property
    def update_anim(self):
        return self.param.animation and (
            (self.ite % self.param.nplot == 0)
            or self.finished)

    @property
    def save_to_file(self):
        if self.param.nhis == 0:
            return False
        else:
            return (self.ite % self.param.nhis == 0) or self.finished
