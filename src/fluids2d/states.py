import numpy as np
from collections import namedtuple


Specs = namedtuple("specs", ("variables", "prognostic"))

model_specs = {
    "euler": Specs(("u", "U", "omega", "ke", "p", "div", "flx"), ("u",)),
    "eulerpsi": Specs(("omega", "U", "psi", "vomega", "flx"), ("omega",)),
    "boussinesq": Specs(("b", "u", "U", "omega", "ke", "p", "div", "flx"), ("b", "u")),
    "hydrostatic": Specs(("b", "uh", "U", "omega", "ke", "p", "div", "flx"), ("b", "uh")),
    "rsw": Specs(("u", "h", "U", "omega", "ke", "p", "flx", "pv"), ("u", "h")),
    "qgrsw": Specs(("u", "h", "U", "omega", "ke", "p", "flx", "pv", "psi"), ("u", "h")),
    "qg": Specs(("pv", "U", "h", "flx", "work", "psi"), ("pv", )),
    "advection": Specs(("q", "U", "flx"), ("q",)),
    "vectoradv": Specs(("v", "U", "omega", "q"), ("v",))
}

vectors = ["u", "U", "flx", "v"]


def get_specs(param):
    specs = model_specs[param.model]
    if (param.tracer is None) | (param.tracer == "None"):
        return specs

    newspecs = add_tracer_to_spec(specs, param.tracer)
    return newspecs


def add_tracer_to_spec(specs, tracer):
    p = specs.prognostic + (tracer,)
    v = p + specs.variables[len(specs.prognostic):]
    return Specs(v, p)


def State(param, shape):
    specs = get_specs(param)
    check_specs_are_well_ordered(specs)
    return allocate_state(param.model, specs.variables, shape)


def Prognostic(param, shape):
    specs = get_specs(param)
    return allocate_state(param.model, specs.prognostic, shape)


def check_specs_are_well_ordered(specs):
    nprognostic = len(specs.prognostic)
    assert specs.variables[:nprognostic] == specs.prognostic


def define_Namedtuple(name, variables):

    class Namedtuple(namedtuple(name, variables)):

        def __repr__(self):
            return f"{name} with {variables}"

    return Namedtuple


Vector = define_Namedtuple("vector", ("x", "y"))


def allocate_var(name, shape):
    if name in vectors:
        return Vector(x=np.zeros(shape),
                      y=np.zeros(shape))
    else:
        return np.zeros(shape)


def allocate_state(name, variables, shape):
    State = define_Namedtuple("state", variables)

    content = {var: allocate_var(var, shape)
               for var in variables}
    return State(**content)


def zero(x): return (np.zeros(x.shape)
                     if isinstance(x, np.ndarray)
                     else
                     x.copy()
                     )


if __name__ == "__main__":
    shape = (5, 3)
    FakedParam = namedtuple("param", ["name", "tracer"])
    param = FakedParam("euler", None)
    s = State(param, shape)
    ds = Prognostic(param, shape)
