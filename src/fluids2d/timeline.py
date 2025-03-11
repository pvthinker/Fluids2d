import numpy as np


class Time:
    def __init__(self, param):
        self.t = 0.0
        self.ite = 0
        self.t0 = 0
        self.ite0 = 0
        self.param = param
        self.dt = param.dt if param.dt > 0 else 0.01
        self._c = 0.0

    @property
    def finished(self):
        return ((self.t >= self.t0+self.param.tend)
                or (self.ite >= self.ite0+self.param.maxite)
                )

    def pushforward(self):
        # instead of doing
        # self.t += self.dt
        #
        # we use the Kahan summation algorithm
        #
        # https://en.wikipedia.org/wiki/Kahan_summation_algorithm
        #
        # this limits the accumulation of truncation errors
        # and makes rounder values for self.t
        y = self.dt - self._c
        t = self.t+y
        self._c = (t-self.t)-y
        self.t = t

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
