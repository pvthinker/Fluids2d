_models = ["euler", "eulerpsi", "advection", "vectoradv",
           "boussinesq", "hydrostatic", "rsw", "qgrsw", "qg"]

_methods = ["weno", "upwind", "centered", "cweno"]
_methods_extended = _methods + ["classic"]

_integrators = ["rk3", "ef", "enrk3", "LFRA"]


class Param:
    def __init__(self):

        self.model = "euler"

        self.nx = 40
        self.ny = 40

        self.Lx = 1.0
        self.Ly = 1.0

        self.xperiodic = False
        self.yperiodic = False
        self.halowidth = 3

        self.f0 = 10.0
        self.g = 1
        self.H = 1

        self.tend = 1.0
        self.maxite = 100
        self.dtmax = 9e99
        self.nprint = 1

        self.nplot = 5
        self.animation = False
        self.plotvar = None
        self.clims = None
        self.cmap = "RdBu_r"

        self.outputfile = "history.nc"
        self.var_to_store = []
        self.nhis = 0

        self.integrator = "rk3"
        self.cfl = 0.9
        self.RAgamma = 0.1

        self.compflux = "weno"
        self.vortexforce = "weno"
        self.innerproduct = "weno"
        self.maxorder = 6

        self.tracer = None

        self.nthreads = 1

        self.__parameters__ = get_parameters(self)
        self.help()

    def add_parameter(self, name):
        setattr(self, name, None)
        self.__parameters__ = get_parameters(self)

    def check_parameters_are_known(self):
        extra = set(get_parameters(self)).difference(set(self.__parameters__))
        if len(extra) == 0:
            return True
        else:
            msg = [f"parameter {extra} is unknown",
                   f"parameters are {self.__parameters__}"]
            assert False, "\n".join(msg)

    def check(self):

        assert self.model in _models

        assert self.compflux in _methods
        assert self.vortexforce in _methods
        assert self.innerproduct in _methods_extended

        assert self.integrator in _integrators

        self.check_parameters_are_known()

    def help(self):
        doc = ["Valid values for string parameters"]
        doc += [f"  - {bold('model')}: " + ", ".join(_models)]
        doc += [f"  - {bold('integrator')}: " + ", ".join(_integrators)]
        doc += [f"  - {bold('compflux')} (U*q): " + ", ".join(_methods)]
        doc += [f"  - {bold('vortexforce')} (omega x U): " +
                ", ".join(_methods)]
        doc += [f"  - {bold('innerproduct')} (U.u): " +
                ", ".join(_methods_extended)]
        doc += [""]
        print("\n".join(doc))


def bold(s):
    return '\033[1m'+'\033[94m'+s+'\033[0m'


def get_parameters(self):
    return [d
            for d in self.__dir__()
            if "__" not in d]
