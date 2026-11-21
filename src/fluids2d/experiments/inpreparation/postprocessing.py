import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

# ------
ncfile = "history.nc"
ds = xr.load_dataset(ncfile)
ds.omega[-1].plot()
