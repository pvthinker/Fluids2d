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

        self.forcing = None
        self.tracer = None

        self.nthreads = 1

    def check(self):

        models = ["euler", "eulerpsi", "advection", "vectoradv",
                  "boussinesq", "hydrostatic", "rsw", "qgrsw", "qg"]
        assert self.model in models

        methods = ["weno", "upwind", "centered"]
        assert self.compflux in methods
        assert self.vortexforce in methods
        assert self.innerproduct in methods+["classic"]