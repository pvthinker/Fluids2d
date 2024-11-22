import unittest
import fluids2d as fd
import experiments.vortex as vortex
import experiments.tracer_advection as tadv


class TestModels(unittest.TestCase):
    def test_euler(self):
        p = fd.Param()
        p.animation = False
        p.tend = 10
        model = fd.Model(p)
        vortex.set_initial_dipole(model)
        model.run()
        self.assertEqual(model.time.ite, 22)

    def test_advection(self):
        p = fd.Param()
        p.model = "advection"
        p.animation = False
        p.tend = 10
        model = fd.Model(p)
        tadv.set_initial_velocity(model, flow="body")
        tadv.set_initial_tracer(model)
        model.run()
        self.assertEqual(model.time.ite, 37)


if __name__ == '__main__':
    unittest.main()
