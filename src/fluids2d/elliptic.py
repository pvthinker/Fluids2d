import numpy as np
from scipy import sparse
import scipy.sparse.linalg as splinalg


class Poisson1d:
    def __init__(self, mesh):
        self.mesh = mesh
        self.A, self.G = self.get_Laplacian()
        self.A_LU = splinalg.splu(self.A)

    def solve(self, b, usingLU=True):
        x = np.zeros_like(b)
        if usingLU:
            x[self.G > -1] = self.A_LU.solve(b[self.G > -1])
        else:
            x[self.G > -1] = sparse.linalg.spsolve(self.A, b[self.G > -1])

        return x

    def get_Laplacian(self):
        nz = self.mesh.ny
        dx = self.mesh.dx
        dz = self.mesh.dy

        xperiodic = self.mesh.param.xperiodic
        n1 = self.mesh.shape[-1]
        nx = self.mesh.nx
        if xperiodic:
            nh = self.mesh.param.halowidth
            G = np.zeros((n1,), dtype="i")-1
            G[nh:nh+nx] = np.arange(nx)
            ileft, iright = nh, nh+nx-1
        else:
            G = np.arange(n1)
            G[-1] = -1
            ileft, iright = 0, n1-1

        n = max(G)+1
        A = np.zeros((n, n))

        coef = np.zeros((n1,))
        for i in range(n1):
            coef[i] = nz/dx**2

        for i in range(n1-1):
            I = G[i]
            if (I > -1):
                Jm = (G[i-1]
                      if i > ileft else
                      (G[-nh-1] if xperiodic else -1)
                      )
                Jp = (G[i+1]
                      if i < iright else
                      (G[nh] if xperiodic else -1)
                      )
                if Jm > -1:
                    A[I, Jm] += coef[i-1]
                    A[I, I] -= coef[i-1]
                if Jp > -1:
                    A[I, Jp] += coef[i]
                    A[I, I] -= coef[i]

        # remove last row/col to make the matrix non singular
        G[G == (n-1)] = -1
        A = sparse.csc_matrix(A[:-1, :-1])

        return A, G


class Poisson2D:
    def __init__(self, mesh, location, maindiag=0):
        assert location in ["c", "v"]
        self.mesh = mesh
        self.location = location
        self.A, self.G = get_Laplacian_sparse(
            mesh, location, maindiag=maindiag)
        self.A_LU = splinalg.splu(self.A)

    def solve(self, b, x, usingLU=True):

        if usingLU:
            x[self.G > -1] = self.A_LU.solve(b[self.G > -1])
        else:
            x[self.G > -1] = sparse.linalg.spsolve(self.A, b[self.G > -1])

        return self.mesh.fill(x)

    def get_rhs(self, config="basic"):
        ny, nx = self.G.shape
        b = np.zeros(self.G.shape)
        if config == "basic":
            b[2*ny//3, nx//3] = 1
            b[2*ny//3, 2*nx//3] = -1

        elif config == "mixed":
            b[ny//3+ny//5, nx//5] = 1
            b[2*ny//3+ny//5, 2*nx//3] = -1
        return b


def get_msk_from_mesh(mesh, location):
    msk = mesh.msk if location == "c" else mesh.mskv
    if not mesh.param.xperiodic:
        return msk

    n = mesh.param.halowidth
    msk_haloed = msk*1
    msk_haloed[:, :n] = 0
    msk_haloed[:, -n:] = 0
    return msk_haloed


def get_Laplacian_sparse(mesh, location, maindiag):
    BCs = {"c": "Neumann",
           "v": "Dirichlet"}

    assert location in BCs

    BC = BCs[location]
    msk = get_msk_from_mesh(mesh, location)
    G = get_Gindex(msk)
    ny, nx = G.shape

    N = np.sum(G > -1)  # max nb of fluid cells

    data = np.zeros((5*N,))  # max nb of elements to store (5 diags)
    row = np.zeros((5*N,), dtype="i")
    col = np.zeros((5*N,), dtype="i")
    counter = 0

    def add_entry(coef, r, c, counter):
        data[counter] = coef
        row[counter] = r
        col[counter] = c
        return counter + 1

    dx2 = mesh.dy/mesh.dx
    dy2 = mesh.dx/mesh.dy

    xperiodic = mesh.param.xperiodic
    yperiodic = False

    n1 = mesh.param.halowidth if xperiodic else 0
    n2 = mesh.param.halowidth if yperiodic else 0

    for j in range(ny):
        for i in range(nx):
            I = G[j, i]
            if I > -1:
                sum_extra_diag = 0

                west = (G[j, i - 1]
                        if i > n1 else
                        (G[j, -n1-1] if xperiodic else -1)
                        )
                east = (G[j, i+1]
                        if i < nx-1-n1 else
                        (G[j, n1] if xperiodic else -1)
                        )
                south = (G[j-1, i]
                         if j > n2 else
                         (G[-n2-1, i] if yperiodic else -1)
                         )
                north = (G[j+1, i]
                         if j < ny-1-n2 else
                         (G[n2, i] if yperiodic else -1)
                         )

                if west > -1:
                    counter = add_entry(dx2, I, west, counter)
                    sum_extra_diag += dx2

                if east > -1:
                    counter = add_entry(dx2, I, east, counter)
                    sum_extra_diag += dx2

                if south > -1:
                    counter = add_entry(dy2, I, south, counter)
                    sum_extra_diag += dy2

                if north > -1:
                    counter = add_entry(dy2, I, north, counter)
                    sum_extra_diag += dy2

                counter = (
                    add_entry(-2*(dx2+dy2)-maindiag, I, I, counter)
                    if BC == "Dirichlet" else
                    add_entry(-sum_extra_diag-maindiag, I, I, counter)
                )

    A = sparse.coo_matrix((data, (row, col)), shape=(N, N))
    # the optimal format for linear algebra is
    # CSC or CSR (and not COO)!
    return A.tocsc(), G


def get_Gindex(msk):
    G = np.zeros(msk.shape, dtype="i")
    nb_fluid_cells = np.sum(msk)
    G[msk == 0] = -1
    G[msk == 1] = np.arange(nb_fluid_cells)
    return G
