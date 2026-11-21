import unittest

import fluids2d as fd
import fluids2d.experiments.tracer_advection as tadv
from fluids2d.experiments import geos_adj, tqg_vortex, vortex


class TestModels(unittest.TestCase):
    def test_euler(self):
        p = fd.Param()
        p.animation = False
        p.tend = 10
        model = fd.Model(p)
        vortex.set_initial_dipole(model)
        model.run()
        # the run must stop because t reaches tend, not maxite
        self.assertTrue(model.time.finished)
        self.assertGreaterEqual(model.time.t, p.tend)
        self.assertLess(model.time.ite, p.maxite)
        # ite is set by the CFL-limited dt, not by tend: 22 iterations
        # with the default cfl=0.9 on a 40x40 mesh (dt ~ 0.46)
        self.assertEqual(model.time.ite, 22)

    def test_advection(self):
        p = fd.Param()
        p.model = "advection"
        p.animation = False
        p.tend = 10
        model = fd.Model(p)
        tadv.set_initial_velocity(model, flow="bodyrotation")
        tadv.set_initial_tracer(model)
        model.run()
        self.assertTrue(model.time.finished)
        self.assertGreaterEqual(model.time.t, p.tend)
        self.assertLess(model.time.ite, p.maxite)
        # deterministic bound given the initial flow and cfl=0.9
        self.assertEqual(model.time.ite, 37)

    def test_tqg(self):
        p = fd.Param()
        p.model = "tqg"
        p.animation = False
        p.tend = 3
        model = fd.Model(p)
        tqg_vortex.set_initial_dipole(model)
        model.run()
        # the run must stop because t reaches tend, not maxite
        self.assertTrue(model.time.finished)
        self.assertGreaterEqual(model.time.t, p.tend)
        self.assertLess(model.time.ite, p.maxite)
        # ite is set by the CFL-limited dt (~0.08, fast dipole): 38 iterations
        # on the default 40x40 mesh with cfl=0.9
        self.assertEqual(model.time.ite, 38)

    def test_rsw(self):
        p = fd.Param()
        p.model = "rsw"
        p.animation = False
        p.tend = 0.5
        model = fd.Model(p)
        geos_adj.set_initial(model, amp=0.2, r0=0.1, flow="dipole")
        model.run()
        # the run must stop because t reaches tend, not maxite
        self.assertTrue(model.time.finished)
        self.assertGreaterEqual(model.time.t, p.tend)
        self.assertLess(model.time.ite, p.maxite)
        # ite is set by the CFL-limited dt (~0.011, fast gravity waves):
        # 45 iterations on the default 40x40 mesh with cfl=0.9
        self.assertEqual(model.time.ite, 45)


if __name__ == '__main__':
    unittest.main()
