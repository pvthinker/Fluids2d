from numba import jit, prange, set_num_threads

set_num_threads(12)

@jit(parallel=True)
def thread_execution(func, arrays, args, nthreads, halo):
    N = arrays[0].size
    n = N//nthreads
    for i in prange(nthreads):
        #idx = slice(max(0,i*n-3*halo),min(i*n+n+2*halo,N))
        i0,i1 = max(0,i*n-3*halo),min(i*n+n+2*halo,N)
        #arr = [a.reshape(-1)[idx] for a in arrays]
        #func(*arr, *args)
        func(*arrays, *args, i0, i1)
